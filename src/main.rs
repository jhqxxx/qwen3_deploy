#[macro_use]
extern crate rocket;

use clap::Parser;
use rocket::Config;
use rocket::data::{ByteUnit, Limits};
use std::io::Write;
use std::{env, fs};

use qwen3_deploy::init;

mod api;
mod utils;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 10100)]
    port: u16,

    #[arg(short, long)]
    model_path: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    write_pid()?;
    start_http_server(args.port, args.model_path).await?;
    Ok(())
}

pub async fn start_http_server(port: u16, model_path: String) -> anyhow::Result<()> {
    let mut builder = rocket::build().configure(Config {
        port,
        limits: Limits::default()
            .limit("string", ByteUnit::Mebibyte(5))
            .limit("json", ByteUnit::Mebibyte(5))
            .limit("data-form", ByteUnit::Mebibyte(100))
            .limit("file", ByteUnit::Mebibyte(100)),
        ..Config::debug_default()
    });

    // 知识库
    builder = builder.mount("/chat", routes![api::chat]);

    init(&model_path)?;

    builder.launch().await?;
    Ok(())
}

fn write_pid() -> anyhow::Result<()> {
    let pid = std::process::id();
    fs::File::create(&env::current_exe()?.parent().unwrap().join(".pid"))?
        .write_all(pid.to_string().as_bytes())?;
    Ok(())
}
