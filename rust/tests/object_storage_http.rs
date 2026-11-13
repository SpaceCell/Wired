// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Object storage read tests against the HTTP backend.
//!
//! Files are served over `http://` by an HTTP/1.1 server on a loopback port
//! within the test process. The server answers `GET` and `HEAD` requests,
//! ranged `GET` requests with `206 Partial Content`, and `404 Not Found` for
//! paths it doesn't serve.

#![cfg(feature = "https")]

mod object_storage_checks;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use lightstream::models::readers::ipc::file_table::FileTableReader;
use lightstream::models::readers::object_storage::StorageReader;
use lightstream::models::writers::ipc::sync_table::write_tables_to_file_sync;
use minarrow::Table;
use object_storage_checks as checks;

/// Serve `files`, keyed by request path, on a loopback port, and return the
/// server's base URL.
fn serve(files: HashMap<String, Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback port binds");
    let base = format!("http://{}", listener.local_addr().unwrap());
    let files = Arc::new(files);
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let files = Arc::clone(&files);
            thread::spawn(move || respond(stream, &files));
        }
    });
    base
}

/// Answer one request on `stream` and close the connection.
fn respond(stream: TcpStream, files: &HashMap<String, Vec<u8>>) {
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    // Read the headers up to the blank line, keeping the inclusive byte
    // range of a `Range: bytes=start-end` header.
    let mut range: Option<(usize, usize)> = None;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) if line == "\r\n" => break,
            Ok(_) => {}
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("range")
        {
            range = value
                .trim()
                .strip_prefix("bytes=")
                .and_then(|bounds| bounds.split_once('-'))
                .map(|(start, end)| (start.parse().unwrap(), end.parse().unwrap()));
        }
    }

    let mut stream = &stream;
    let Some(file) = files.get(&path) else {
        let _ = write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        return;
    };
    let (status, content_range, body) = match range {
        Some((start, end)) => {
            let end = end.min(file.len() - 1);
            (
                "206 Partial Content",
                format!("Content-Range: bytes {start}-{end}/{}\r\n", file.len()),
                &file[start..=end],
            )
        }
        None => ("200 OK", String::new(), &file[..]),
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{content_range}Connection: close\r\n\r\n",
        body.len()
    );
    if method != "HEAD" {
        let _ = stream.write_all(body);
    }
}

/// Arrow IPC file bytes holding two tables with ids 0 to 14.
fn ipc_file_bytes() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trades.arrow");
    let first = checks::sample_table(0, 10);
    let schema = checks::schema_of(&first);
    write_tables_to_file_sync(&path, &[first, checks::sample_table(10, 5)], schema).unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn storage_reader_fetches_a_whole_file_over_http() {
    let contents: Vec<u8> = (0..200_000u32).flat_map(|i| i.to_le_bytes()).collect();
    let base = serve(HashMap::from([("/data/a.bin".to_string(), contents.clone())]));
    let runtime = checks::runtime();
    let _entered = runtime.enter();

    let mut read = Vec::new();
    StorageReader::open(format!("{base}/data/a.bin"))
        .unwrap()
        .read_to_end(&mut read)
        .unwrap();
    assert!(read == contents, "file contents match what was served");
}

#[test]
fn ipc_file_reader_reads_batches_with_ranged_requests_over_http() {
    let base = serve(HashMap::from([(
        "/ipc/trades.arrow".to_string(),
        ipc_file_bytes(),
    )]));
    let runtime = checks::runtime();
    let _entered = runtime.enter();

    let reader = FileTableReader::open(format!("{base}/ipc/trades.arrow")).unwrap();
    assert_eq!(reader.num_batches(), 2);
    let tables: Vec<Table> = (0..2).map(|i| reader.read_batch(i).unwrap()).collect();
    assert_eq!(checks::ids(&tables), (0..15).collect::<Vec<_>>());
}

#[test]
fn a_missing_path_reports_not_found() {
    let base = serve(HashMap::new());
    let runtime = checks::runtime();
    let _entered = runtime.enter();
    checks::a_missing_object_reports_not_found(&base);
}
