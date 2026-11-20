// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! WebTransport example server streaming the demo table as Arrow IPC.
//!
//! Serves the same table as server.py, so the Python client reads
//! identical results from either backend.
//!
//! Run with:
//! ```sh
//! cargo run --example wt_server
//! ```

#[path = "../../common/args.rs"]
mod args;
#[path = "../../common/datagen.rs"]
mod datagen;
#[path = "../../common/tls.rs"]
mod tls;

use std::io;

use lightstream::models::transports::webtransport::WebTransport;
use lightstream::models::writers::webtransport::WebTransportTableWriter;
use lightstream::traits::transport_writer::IPCTransportWriter;
use tokio::runtime::Runtime;

async fn serve() -> io::Result<()> {
    let uri = args::example_uri("wt://127.0.0.1:9046");
    let addr = args::authority(&uri)
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let config = tls::wt_server(addr, &args::tls_cert(), &args::tls_key()).await?;
    // WebTransport sessions are requested over HTTPS. The endpoint URL
    // therefore uses the `https` scheme with the URI's authority and path.
    let rest = uri.split_once("://").map_or(uri.as_str(), |(_, rest)| rest);
    let listener = WebTransport::bind(&format!("https://{rest}"), config)?;
    let table = datagen::get_table();
    let schema = datagen::schema(&table);
    println!("Serving get_table on {uri}");

    loop {
        let (_recv, send) = WebTransport::accept(&listener).await?;
        let mut writer = WebTransportTableWriter::new(send, schema.clone(), None)?;
        writer.write_table(table.clone()).await?;
        writer.finish().await?;
    }
}

fn main() -> io::Result<()> {
    Runtime::new()?.block_on(serve())
}
