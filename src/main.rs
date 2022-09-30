use actix_web::{get, App, HttpResponse, HttpServer, Responder};
// use tract_ndarray::Array;
use tract_onnx::prelude::*;
use rand::*;


fn predict() -> TractResult<()> {
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("model.onnx")?
        // specify input type and shape
        .with_input_fact(0, f64::fact(&[1, 20]).into())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // Generate some input data for the model
    let mut rng = thread_rng();
    let vals: Vec<_> = (0..20).map(|_| rng.gen::<f64>()).collect();
    let input = tract_ndarray::arr1(&vals).into_shape((1, 20)).unwrap();

    // Input the generated data into the model
    let result = model.run(tvec![input.into()]).unwrap();
    let to_show = result[0].to_array_view::<i64>()?;
    println!("result: {:?}", to_show);
    Ok(())

}

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok().body("index")
}

#[get("/healthcheck")]
async fn healthcheck() -> impl Responder {
    HttpResponse::Ok().body("Ok")
}

#[get("/prediction")]
async fn prediction() -> impl Responder {
    HttpResponse::Ok().body("test")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {

    HttpServer::new(|| {
        App::new()
            .service(index)
            .service(healthcheck)
            .service(prediction)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}