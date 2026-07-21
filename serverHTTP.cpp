#define _WIN32_WINNT 0x0A00
#include <iostream>
#include "external/httplib.h"
#include "external/json.hpp"
#include <tokenizers_cpp.h>
#include <chrono>
#include "external/LoadBytesFromFile.cpp"
#include "inference_request.h"
#include "serverHTTP.h"
#include "server_queue.h"
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



	ServerQueue queue;

	svr.Post("/prompt", [tokenizer, &queue](const auto &req, auto &res){
		try{
			auto j = json::parse(req.body);

			if (!j.contains("prompt")){
				res.status = 400;
				res.set_content("{\"error\": \"Se requiere campo 'prompt'\"}", "application/json");
				return;
			}

			std::string prompt = j["prompt"];
			std::vector<int> ids = tokenizer->Encode(prompt);
			std::cout << "Prompt recibido: " << prompt << std::endl;

			InferenceRequest request;
			request.prompt = prompt;
			request.tokens = std::move(ids); // move semantics to contain ids in tokens

			if (j.contains("temperature")) {
				request.temperature = j["temperature"];
			}
			if (j.contains("max_quantity_of_tokens")) {
				request.max_quantity_of_tokens = j["max_quantity_of_tokens"];
			}

			uint64_t id = queue.post(std::move(request)); // move semantics again

			json response = {
				{"id", id},
				{"status", "queued"}
			};
			res.set_content(response.dump(), "application/json");

		}catch(const std::exception& d){
			res.status = 400;
			res.set_content("{\"error\": \"JSON inválido\"}", "application/json");
		}
	});

	
	svr.listen("0.0.0.0", 8080);
}



