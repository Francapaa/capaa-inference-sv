#define _WIN32_WINNT 0x0A00
#include <iostream>
#include "external/httplib.h"
#include "external/json.hpp"
#include <tokenizers_cpp.h>
#include <chrono>
#include "external/LoadBytesFromFile.cpp"
#include "inference_request.h"
#include "serverHTTP.h"
using json = nlohmann::json; 
using namespace httplib;

//primer paso , definir lo que va a venir como request

/*
	PROMPT ==> HTTP (POST)
	PROMPT ==> TOKENS 
	TOKENS (EL PROMPT) ==> REQUEST QUEUE
	REQUEST QUEUE ==> k-v caching (CPU usage / RAM usage)

	SCHEDULER ==> CEREBRO VA A ESTAR MONITOREANDO EL CPU USAGE Y EL PROMPT PARA VER SI HAY ESPACIO PARA 
				  ENVIAR EL PROMPT TOKENIZADO

*/

void run_server(){

	Server svr;

	auto blob = LoadBytesFromFile("C:/Users/franc/OneDrive/Desktop/inference-server/models/tokenizer.json");
	if (blob.empty()) { // daba error por el PATH, necesita el path ABSOLUTO
		std::cerr << "ERROR EN LA FUNCION BLOB NO IDENTIFICO NADA" << std::endl;
		return;
	}

	std::string json_str(blob.data(), blob.size());
	json_str.erase(std::remove_if(json_str.begin(), json_str.end(), [](unsigned char c) {
		return (c < 32 && c != '\n' && c != '\r' && c != '\t');
		}), json_str.end());

	auto tokenizer = std::shared_ptr<tokenizers::Tokenizer>(
		tokenizers::Tokenizer::FromBlobJSON(json_str).release()
	);
															

	svr.Get("/hi", [](const httplib::Request &req, httplib::Response &res) {
	  res.set_content("Hello World!", "text/plain"); // endpoint test
	});



	svr.Post("/prompt", [tokenizer](const auto &req, auto &res){
		try{
			auto j = json::parse(req.body);
			if(j.contains("id") || j.contains("tokens") || j.contains("max_quantity_of_tokens"	) || j.contains("isFinished") ) {
				res.status=400; 
				res.set_content("Error estos datos no son validos", "text/plain"); 
				return; 
			}
			if (j.contains("prompt")){
				std::string prompt = j["prompt"];
				std::vector<int> ids = tokenizer -> Encode(prompt); 
				std::cout<<"Prompt recibido: "<<prompt<<std::endl; 
				
				//creamos el inference request

				auto request = std::make_shared<InferenceRequest>(); 
				//request->id = generate_id(); aca tenemos que crear la funcion que va a generar el id
				request->prompt = prompt; 
				request->tokens = ids; 
				request->enqueue_time = std::chrono::steady_clock::now(); // a partir de ahora se agrega a la cola

				res.set_content("Recibido correctamente", "text/plain");
			}else{
				res.status = 400;
				res.set_content("Ha ocurrido un error", "text/plain"); 
				return; 
			}
		}catch(const std::exception d){
			res.status = 400;
            res.set_content("JSON inválido", "text/plain");
			return; 
		}
	});

	
	svr.listen("0.0.0.0", 8080);
}



