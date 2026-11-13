// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # WebSocket byte stream adapters
//!
//! Provides [`WsRead`] and [`WsWrite`] for WebSocket I/O over a raw TCP
//! stream extracted after the tungstenite handshake.
//!
//! The adapters perform the IO. The WebSocket protocol is held by
//! [`WsFrameDecoder`] on the read side and [`WsFrameEncoder`] on the write
//! side, which work on bytes in memory and don't perform any IO. Payload
//! bytes flow between the TCP socket and the caller's buffer via the
//! standard [`AsyncRead`] / [`AsyncWrite`] traits, with zero intermediate
//! allocations on the read side.
//!
//! [`WsRead`] also implements [`Stream`] yielding arena-backed [`SharedBuffer`]
//! windows for consumers that prefer the StreamExt API.

use std::io;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, ready};

use futures_core::Stream;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use minarrow::structs::shared_buffer::SharedBuffer;

use crate::models::decoders::websocket::{WsDecodeResult, WsFrameDecoder};
use crate::models::encoders::websocket::WsFrameEncoder;
use crate::models::streams::stream_arena::StreamArena;

// ---------------------------------------------------------------------------
// Read path
// ---------------------------------------------------------------------------

/// WebSocket read adapter over a raw TCP stream.
///
/// Parses WS frame headers in a fixed stack buffer and exposes unmasked
/// binary payload bytes via [`AsyncRead`]. Ping and close frames are
/// handled transparently - pongs are sent immediately via the shared
/// write half, matching standard WebSocket library behaviour.
///
/// Also implements [`Stream`] yielding arena-backed [`SharedBuffer`]
/// windows for zero-allocation streaming.
///
/// Frame handling is done by [`WsFrameDecoder::decode_next`], which rejects
/// text frames with `io::ErrorKind::InvalidData`.
pub struct WsRead<R, W> {
    inner: R,
    writer: Arc<Mutex<W>>,
    decoder: WsFrameDecoder,
    arena: StreamArena,
    chunk_size: usize,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> WsRead<R, W> {
    /// Wrap a raw TCP read half as a WebSocket reader.
    ///
    /// The `writer` is the shared write half used for sending pong
    /// responses to pings. The WebSocket handshake must already be complete.
    pub fn new(inner: R, writer: Arc<Mutex<W>>) -> Self {
        Self::with_decoder(inner, writer, WsFrameDecoder::new())
    }

    /// Wrap a raw TCP read half as a client-side WebSocket reader.
    ///
    /// Identical to [`WsRead::new`] except pong responses are masked,
    /// as RFC 6455 requires of every client-to-server frame.
    pub fn new_client(inner: R, writer: Arc<Mutex<W>>) -> Self {
        Self::with_decoder(inner, writer, WsFrameDecoder::new_client())
    }

    fn with_decoder(inner: R, writer: Arc<Mutex<W>>, decoder: WsFrameDecoder) -> Self {
        Self {
            inner,
            writer,
            decoder,
            arena: StreamArena::new(),
            chunk_size: 64 * 1024,
        }
    }
}

/// Read transport bytes into the region the decoder returns for `target`,
/// and return the number of payload bytes delivered into `target`.
///
/// Marks the decoder closed at end of input.
fn poll_fill<R: AsyncRead + Unpin>(
    inner: &mut R,
    decoder: &mut WsFrameDecoder,
    cx: &mut Context<'_>,
    target: &mut [MaybeUninit<u8>],
) -> Poll<io::Result<usize>> {
    let mut read_buf = ReadBuf::uninit(decoder.spare(target));
    ready!(Pin::new(inner).poll_read(cx, &mut read_buf))?;
    let n = read_buf.filled().len();
    if n == 0 {
        decoder.close();
        return Poll::Ready(Ok(0));
    }
    // SAFETY: ReadBuf reports exactly the bytes initialised within the
    // region, and `target` is the buffer passed to `spare`.
    Poll::Ready(Ok(unsafe { decoder.advance(n, target) }))
}

/// Write the decoder's pending pong frame through the shared write half.
fn poll_send_pong<W: AsyncWrite + Unpin>(
    writer: &Mutex<W>,
    decoder: &mut WsFrameDecoder,
    cx: &mut Context<'_>,
) -> Poll<io::Result<()>> {
    let mut guard = writer.lock().unwrap();
    let n = ready!(Pin::new(&mut *guard).poll_write(cx, decoder.pong()))?;
    decoder.advance_pong(n);
    Poll::Ready(Ok(()))
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> AsyncRead for WsRead<R, W> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let me = self.get_mut();

        loop {
            match me.decoder.decode_next()? {
                WsDecodeResult::NeedMore => {
                    // Payload reads into a full buffer return without reading.
                    // An empty region would otherwise read as end of input.
                    if me.decoder.payload_remaining() > 0 && buf.remaining() == 0 {
                        return Poll::Ready(Ok(()));
                    }
                    // SAFETY: the decoder and the transport only write
                    // initialised bytes into the unfilled region.
                    let target = unsafe { buf.unfilled_mut() };
                    let n = ready!(poll_fill(&mut me.inner, &mut me.decoder, cx, target))?;
                    if n > 0 {
                        // SAFETY: `poll_fill` initialised the first `n`
                        // unfilled bytes with unmasked payload.
                        unsafe { buf.assume_init(n) };
                        buf.advance(n);
                        return Poll::Ready(Ok(()));
                    }
                }
                WsDecodeResult::Pong => ready!(poll_send_pong(&me.writer, &mut me.decoder, cx))?,
                WsDecodeResult::Closed | WsDecodeResult::Message(_) => {
                    return Poll::Ready(Ok(()));
                }
            }
        }
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Stream for WsRead<R, W> {
    type Item = Result<SharedBuffer, io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = self.get_mut();

        loop {
            if me.arena.remaining() < me.chunk_size {
                me.arena.recycle_or_reset();
            }

            match me.decoder.decode_next() {
                Err(e) => return Poll::Ready(Some(Err(e))),
                Ok(WsDecodeResult::NeedMore) => {
                    let chunk_start = me.arena.write_pos();
                    let target = me.arena.spare_uninit();
                    let n = match ready!(poll_fill(&mut me.inner, &mut me.decoder, cx, target)) {
                        Ok(n) => n,
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    };
                    if n > 0 {
                        // SAFETY: `poll_fill` initialised the first `n` bytes
                        // of the spare region with unmasked payload.
                        unsafe { me.arena.advance(n) };
                        let shared = me.arena.window(chunk_start, n);
                        me.arena.align();
                        return Poll::Ready(Some(Ok(shared)));
                    }
                }
                Ok(WsDecodeResult::Pong) => {
                    if let Err(e) = ready!(poll_send_pong(&me.writer, &mut me.decoder, cx)) {
                        return Poll::Ready(Some(Err(e)));
                    }
                }
                Ok(WsDecodeResult::Closed | WsDecodeResult::Message(_)) => {
                    return Poll::Ready(None);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Write path
// ---------------------------------------------------------------------------

/// WebSocket write adapter over a raw TCP stream.
///
/// Wraps each `poll_write` call's bytes in a WS binary frame. The WS
/// header is written from a stack buffer, then the caller's payload
/// follows. No intermediate copies.
///
/// Designed for use with `write_all` where each call represents a
/// complete message.
pub struct WsWrite<W> {
    inner: Arc<Mutex<W>>,
    /// Role-dependent framing, including client frame masking, and the
    /// progress of the frame being written.
    encoder: WsFrameEncoder,
}

impl<W: AsyncWrite + Unpin> WsWrite<W> {
    /// Create a server-side WebSocket write adapter from a raw TCP
    /// write half. Frames are written unmasked, as RFC 6455 requires of
    /// server-to-client traffic.
    ///
    /// Returns the shared writer reference needed by WsRead for pong
    /// responses, and the WsWrite adapter for application data.
    /// The WebSocket handshake must already be complete.
    pub fn new(inner: W) -> (Arc<Mutex<W>>, Self) {
        let shared = Arc::new(Mutex::new(inner));
        let writer = Self {
            inner: Arc::clone(&shared),
            encoder: WsFrameEncoder::new(),
        };
        (shared, writer)
    }

    /// Create a client-side WebSocket write adapter from a raw TCP
    /// write half. Every frame is masked with a fresh key, as RFC 6455
    /// requires of client-to-server traffic; compliant servers reject
    /// unmasked client frames.
    pub fn new_client(inner: W) -> (Arc<Mutex<W>>, Self) {
        let (shared, mut writer) = Self::new(inner);
        writer.encoder = WsFrameEncoder::new_client();
        (shared, writer)
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for WsWrite<W> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        let mut guard = me.inner.lock().unwrap();

        // Frames left incomplete by an earlier `Pending` resume from where
        // they stopped.
        if me.encoder.is_complete() {
            me.encoder.begin_binary(buf);
        }
        while !me.encoder.is_complete() {
            let n = ready!(Pin::new(&mut *guard).poll_write(cx, me.encoder.pending(buf)))?;
            me.encoder.advance(n);
        }
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut guard = self.get_mut().inner.lock().unwrap();
        Pin::new(&mut *guard).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        let mut guard = me.inner.lock().unwrap();
        me.encoder.begin_close();
        if let Poll::Ready(Ok(n)) = Pin::new(&mut *guard).poll_write(cx, me.encoder.pending(&[])) {
            me.encoder.advance(n);
        }
        Pin::new(&mut *guard).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::AsyncWriteExt;

    use super::*;

    /// A client-role writer emits RFC 6455 masked frames, i.e.,
    /// mask bit set, key present, and the payload is XORed.
    ///
    /// Interoperability with compliant servers depends on this,
    /// so the assertion is byte-level.
    #[tokio::test]
    async fn client_frames_are_masked() {
        let (tx, mut rx) = tokio::io::duplex(1024);
        let (_shared, mut ws_write) = WsWrite::new_client(tx);

        let payload = b"orderbook";
        ws_write.write_all(payload).await.unwrap();
        drop(ws_write);

        let mut wire = vec![0u8; 2 + 4 + payload.len()];
        tokio::io::AsyncReadExt::read_exact(&mut rx, &mut wire)
            .await
            .unwrap();

        assert_eq!(wire[0], 0x82, "FIN + binary opcode");
        assert_eq!(wire[1] & 0x80, 0x80, "mask bit must be set");
        assert_eq!(wire[1] & 0x7F, payload.len() as u8);
        let key = [wire[2], wire[3], wire[4], wire[5]];
        let unmasked: Vec<u8> = wire[6..]
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ key[i % 4])
            .collect();
        assert_eq!(&unmasked, payload);
    }

    /// A server-role writer stays unmasked, per RFC 6455.
    #[tokio::test]
    async fn server_frames_are_unmasked() {
        let (tx, mut rx) = tokio::io::duplex(1024);
        let (_shared, mut ws_write) = WsWrite::new(tx);

        let payload = b"orderbook";
        ws_write.write_all(payload).await.unwrap();
        drop(ws_write);

        let mut wire = vec![0u8; 2 + payload.len()];
        tokio::io::AsyncReadExt::read_exact(&mut rx, &mut wire)
            .await
            .unwrap();
        assert_eq!(wire[1] & 0x80, 0, "server frames must not be masked");
        assert_eq!(&wire[2..], payload);
    }

    /// Masked client frames decode through the server-side reader.
    #[tokio::test]
    async fn masked_round_trip() {
        let (client_end, server_end) = tokio::io::duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_end);
        let (server_read, server_write) = tokio::io::split(server_end);

        let (_c_shared, mut ws_write) = WsWrite::new_client(client_write);
        let (s_shared, _s_write) = WsWrite::new(server_write);
        let mut ws_read = WsRead::new(server_read, s_shared);
        drop(client_read);

        let payload = vec![0xA5u8; 300];
        let sent = payload.clone();
        let writer = tokio::spawn(async move {
            ws_write.write_all(&sent).await.unwrap();
        });

        let mut received = vec![0u8; payload.len()];
        tokio::io::AsyncReadExt::read_exact(&mut ws_read, &mut received)
            .await
            .unwrap();
        writer.await.unwrap();
        assert_eq!(received, payload);
    }

    #[tokio::test]
    async fn oversized_ping_is_rejected_before_allocation() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(tokio::io::sink());
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x89, 126, 0, 126]).await.unwrap();

        let mut byte = [0u8; 1];
        let error = tokio::io::AsyncReadExt::read_exact(&mut reader, &mut byte)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn pong_payload_is_consumed_before_next_data_frame() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(tokio::io::sink());
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x8A, 3, b'a', b'b', b'c', 0x82, 2, b'o', b'k'])
            .await
            .unwrap();

        let mut payload = [0u8; 2];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut payload)
            .await
            .unwrap();
        assert_eq!(&payload, b"ok");
    }

    /// Reads `len` bytes of pong output written through the shared half.
    async fn read_pong(rx: &mut tokio::io::DuplexStream, len: usize) -> Vec<u8> {
        let mut pong = vec![0u8; len];
        tokio::io::AsyncReadExt::read_exact(rx, &mut pong)
            .await
            .unwrap();
        pong
    }

    /// Server-role readers answer masked pings with unmasked pongs that echo
    /// the ping payload, then read the next data frame.
    #[tokio::test]
    async fn masked_ping_is_answered_with_unmasked_pong() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (pong_tx, mut pong_rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(pong_tx);
        let mut reader = WsRead::new(rx, shared);
        let key = [1u8, 2, 3, 4];
        let masked: Vec<u8> = b"abc".iter().enumerate().map(|(i, b)| b ^ key[i % 4]).collect();
        let mut wire = vec![0x89, 0x83];
        wire.extend_from_slice(&key);
        wire.extend_from_slice(&masked);
        wire.extend_from_slice(&[0x82, 2, b'o', b'k']);
        tx.write_all(&wire).await.unwrap();

        let mut payload = [0u8; 2];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut payload)
            .await
            .unwrap();
        assert_eq!(&payload, b"ok");
        assert_eq!(read_pong(&mut pong_rx, 5).await, [0x8A, 3, b'a', b'b', b'c']);
    }

    /// Client-role readers answer pings with masked pongs.
    #[tokio::test]
    async fn client_pong_is_masked() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (pong_tx, mut pong_rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new_client(pong_tx);
        let mut reader = WsRead::new_client(rx, shared);
        tx.write_all(&[0x89, 2, b'h', b'i', 0x82, 1, b'x']).await.unwrap();

        let mut payload = [0u8; 1];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut payload)
            .await
            .unwrap();
        assert_eq!(&payload, b"x");
        let pong = read_pong(&mut pong_rx, 8).await;
        assert_eq!(pong[0], 0x8A);
        assert_eq!(pong[1], 0x82, "mask bit set with a 2-byte payload");
        let key = [pong[2], pong[3], pong[4], pong[5]];
        assert_eq!([pong[6] ^ key[0], pong[7] ^ key[1]], *b"hi");
    }

    /// Empty pings are answered with empty pongs.
    #[tokio::test]
    async fn empty_ping_is_answered_with_empty_pong() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (pong_tx, mut pong_rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(pong_tx);
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x89, 0, 0x82, 1, b'x']).await.unwrap();

        let mut payload = [0u8; 1];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut payload)
            .await
            .unwrap();
        assert_eq!(read_pong(&mut pong_rx, 2).await, [0x8A, 0]);
    }

    /// Text frames are rejected as unsupported.
    #[tokio::test]
    async fn text_frame_is_rejected() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(tokio::io::sink());
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x81, 2, b'h', b'i']).await.unwrap();

        let mut byte = [0u8; 1];
        let error = tokio::io::AsyncReadExt::read_exact(&mut reader, &mut byte)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    /// Close frames end the byte stream, and later reads stay at end of
    /// stream.
    #[tokio::test]
    async fn close_frame_ends_the_stream() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(tokio::io::sink());
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x82, 2, b'o', b'k', 0x88, 2, 0x03, 0xE8, 0x82, 1, b'x'])
            .await
            .unwrap();

        let mut received = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut received)
            .await
            .unwrap();
        assert_eq!(received, b"ok");
        let mut byte = [0u8; 1];
        let n = tokio::io::AsyncReadExt::read(&mut reader, &mut byte)
            .await
            .unwrap();
        assert_eq!(n, 0);
    }

    /// The `Stream` path yields unmasked payload windows across fragments
    /// and ends at the close frame.
    #[tokio::test]
    async fn stream_yields_unmasked_payload_windows() {
        let (mut tx, rx) = tokio::io::duplex(1024);
        let (shared, _writer) = WsWrite::new(tokio::io::sink());
        let mut reader = WsRead::new(rx, shared);
        let key = [0xA1u8, 0xB2, 0xC3, 0xD4];
        let payload: Vec<u8> = (0u8..200).collect();
        let mut wire = Vec::new();
        // First fragment, 7 bytes, masked.
        wire.extend_from_slice(&[0x02, 0x87]);
        wire.extend_from_slice(&key);
        wire.extend(payload[..7].iter().enumerate().map(|(i, b)| b ^ key[i % 4]));
        // Pong between fragments, then the final masked continuation.
        wire.extend_from_slice(&[0x8A, 1, 0xFF]);
        wire.extend_from_slice(&[0x80, 0x80 | 126, 0, 193]);
        wire.extend_from_slice(&key);
        wire.extend(payload[7..].iter().enumerate().map(|(i, b)| b ^ key[i % 4]));
        wire.extend_from_slice(&[0x88, 0]);
        tx.write_all(&wire).await.unwrap();

        let mut received = Vec::new();
        while let Some(window) = futures_util::StreamExt::next(&mut reader).await {
            received.extend_from_slice(window.unwrap().as_slice());
        }
        assert_eq!(received, payload);
    }

    /// End of input while a frame header is incomplete ends the stream.
    #[tokio::test]
    async fn eof_inside_header_ends_the_stream() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(tokio::io::sink());
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x82, 126, 0]).await.unwrap();
        drop(tx);

        let mut received = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut received)
            .await
            .unwrap();
        assert!(received.is_empty());
    }

    /// Unmasks client frame payload bytes with the key that precedes them.
    fn unmask_with(key: [u8; 4], masked: &[u8]) -> Vec<u8> {
        masked.iter().enumerate().map(|(i, b)| b ^ key[i % 4]).collect()
    }

    /// Server-role writers frame each write as one unmasked binary frame,
    /// with 16-bit and 64-bit extended lengths above 125 and 65535 bytes.
    #[tokio::test]
    async fn server_frames_use_extended_lengths() {
        let (tx, mut rx) = tokio::io::duplex(1 << 20);
        let (_shared, mut ws_write) = WsWrite::new(tx);
        let medium = vec![0x11u8; 300];
        let large = vec![0x22u8; 70_000];
        ws_write.write_all(&medium).await.unwrap();
        ws_write.write_all(&large).await.unwrap();
        drop(ws_write);
        drop(_shared);

        let mut wire = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut rx, &mut wire)
            .await
            .unwrap();
        let mut expected = vec![0x82, 126];
        expected.extend_from_slice(&300u16.to_be_bytes());
        expected.extend_from_slice(&medium);
        expected.extend_from_slice(&[0x82, 127]);
        expected.extend_from_slice(&70_000u64.to_be_bytes());
        expected.extend_from_slice(&large);
        assert_eq!(wire, expected);
    }

    /// Client-role writers mask each frame with its own key, with the same
    /// extended lengths as server frames.
    #[tokio::test]
    async fn client_frames_use_extended_lengths() {
        let (tx, mut rx) = tokio::io::duplex(1 << 20);
        let (_shared, mut ws_write) = WsWrite::new_client(tx);
        let medium: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        let large: Vec<u8> = (0..70_000u32).map(|i| (i * 7) as u8).collect();
        ws_write.write_all(&medium).await.unwrap();
        ws_write.write_all(&large).await.unwrap();
        drop(ws_write);
        drop(_shared);

        let mut wire = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut rx, &mut wire)
            .await
            .unwrap();
        assert_eq!(&wire[..2], &[0x82, 0x80 | 126]);
        assert_eq!(&wire[2..4], &300u16.to_be_bytes());
        let key = [wire[4], wire[5], wire[6], wire[7]];
        assert_eq!(unmask_with(key, &wire[8..308]), medium);
        let rest = &wire[308..];
        assert_eq!(&rest[..2], &[0x82, 0x80 | 127]);
        assert_eq!(&rest[2..10], &70_000u64.to_be_bytes());
        let key = [rest[10], rest[11], rest[12], rest[13]];
        assert_eq!(unmask_with(key, &rest[14..]), large);
    }

    /// Server-role writers send an unmasked normal-closure close frame on
    /// shutdown.
    #[tokio::test]
    async fn server_shutdown_sends_close_frame() {
        let (tx, mut rx) = tokio::io::duplex(64);
        let (_shared, mut ws_write) = WsWrite::new(tx);
        ws_write.shutdown().await.unwrap();

        let mut wire = [0u8; 4];
        tokio::io::AsyncReadExt::read_exact(&mut rx, &mut wire)
            .await
            .unwrap();
        assert_eq!(wire, [0x88, 2, 0x03, 0xE8]);
    }

    /// Client-role writers send a masked normal-closure close frame on
    /// shutdown.
    #[tokio::test]
    async fn client_shutdown_sends_masked_close_frame() {
        let (tx, mut rx) = tokio::io::duplex(64);
        let (_shared, mut ws_write) = WsWrite::new_client(tx);
        ws_write.shutdown().await.unwrap();

        let mut wire = [0u8; 8];
        tokio::io::AsyncReadExt::read_exact(&mut rx, &mut wire)
            .await
            .unwrap();
        assert_eq!(&wire[..2], &[0x88, 0x82]);
        let key = [wire[2], wire[3], wire[4], wire[5]];
        assert_eq!(unmask_with(key, &wire[6..]), [0x03, 0xE8]);
    }

    /// Zero-capacity reads answer pings ahead of a data frame and return
    /// without closing, and the next read with capacity returns the payload.
    #[tokio::test]
    async fn zero_capacity_read_answers_ping_without_closing() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let (pong_tx, mut pong_rx) = tokio::io::duplex(64);
        let (shared, _writer) = WsWrite::new(pong_tx);
        let mut reader = WsRead::new(rx, shared);
        tx.write_all(&[0x89, 1, b'p', 0x82, 2, b'o', b'k']).await.unwrap();

        let n = tokio::io::AsyncReadExt::read(&mut reader, &mut []).await.unwrap();
        assert_eq!(n, 0);
        assert_eq!(read_pong(&mut pong_rx, 3).await, [0x8A, 1, b'p']);

        let mut payload = [0u8; 2];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut payload)
            .await
            .unwrap();
        assert_eq!(&payload, b"ok");
    }
}
