use qwen3_deploy::{chat_stream, chat_sync, ChatRequest};
use rocket::Request;
use rocket::futures::{Stream, StreamExt};
use rocket::http::{ContentType, Status};
use rocket::response::Responder;
use rocket::response::stream::TextStream;
use rocket::serde::json::Json;

enum Response<R: Stream<Item = String> + Send> {
    Stream(TextStream<R>),
    Text(String),
    Error(String),
}

impl<'r, 'o: 'r, R> Responder<'r, 'o> for Response<R>
where
    R: Stream<Item = String> + Send + 'o,
    'r: 'o,
{
    fn respond_to(self, req: &'r Request<'_>) -> rocket::response::Result<'o> {
        match self {
            Response::Stream(stream) => stream.respond_to(req),
            Response::Text(text) => text.respond_to(req),
            Response::Error(e) => {
                let mut res = rocket::response::Response::new();
                res.set_status(Status::InternalServerError);
                res.set_header(ContentType::JSON);
                res.set_sized_body(e.len(), std::io::Cursor::new(e));
                Ok(res)
            }
        }
    }
}

#[post("/completions", data = "<req>")]
pub(crate) async fn chat(req: Json<ChatRequest>) -> (ContentType, Response<impl Stream<Item = String>>) {
    match req.stream {
        Some(false) => {
            match chat_sync(&req.into_inner()).await {
                Ok(response) => {
                    (ContentType::Text, Response::Text(response))
                }
                Err(e) => {
                     (ContentType::Text, Response::Error(e.to_string()))
                }
            }
        },
        _ => {
            let stream = TextStream! {
                match chat_stream(&req.into_inner()) {
                    Ok(stream) => {
                        let mut boxed_stream = Box::pin(stream);
                        while let Some(resp) = boxed_stream.next().await {
                            yield format!("event: message\ndata: {}\n\n", resp);
                        }
                        yield format!("event: message\ndata: {}\n\n", "[DONE]");
                    }
                    Err(e) => {
                        yield format!("event: error\ndata: {}\n\n", e.to_string());
                    }
                }
            };
            (ContentType::EventStream, Response::Stream(stream))
        }
    }
    
}
