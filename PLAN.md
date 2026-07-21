# Inference Server — Plan de desarrollo

## Stack

```
Cliente ──HTTP──> Servidor ──Queue──> Scheduler ──Engine──> Model + KV-cache
                        (httplib)    (propio)      (propio)    (desde cero)
```

---

## Fase 0 — Ordenar la casa

| # | Tarea | Descripción | Estado |
|---|-------|-------------|--------|
| 0.1 | `inference_request.h` | Struct `InferenceRequest` compartido entre todos los módulos | ✅ |
| 0.2 | Separar `main()` | `main()` → `main.cpp`, server logic → `serverHTTP.cpp` + `serverHTTP.h` | ✅ |
| 0.3 | `server_queue.cpp` | Arreglar errores: tipo `server_task` indefinido, mutex sin lock, `id_tasks` inexistente, syntax error | ✅ |
| 0.4 | `CMakeLists.txt` | Agregar `server_queue.cpp` al build | ✅ |
| 0.5 | Endpoint `/prompt` JSON | Devuelve `{"id": N, "status": "queued"}` | ✅ |

---

## Fase 1 — Motor de tensores + tiny model

| # | Tarea | Descripción |
|---|-------|-------------|
| 1.1 | `Tensor` | Wrapper sobre `std::vector<float>` con shape, strides, slicing |
| 1.2 | Operaciones | `matmul`, `add`, `softmax`, `layer_norm`, `silu` |
| 1.3 | `Linear` | Capa fully-connected: `y = xW^T + b` |
| 1.4 | **Bi-gram model** | Tabla de frecuencias `vocab_size × vocab_size`. Sin atención, sin transformer. |
| 1.5 | Loop de generación | predict → sample → repeat hasta `max_quantity_of_tokens` o EOS |
| 1.6 | Conectar modelo → servidor | Worker desencola, ejecuta generación, devuelve respuesta |

**Resultado**: `POST /prompt` genera texto (mala calidad, pero pipeline completo). ✅

---

## Fase 2 — Transformer didáctico

| # | Tarea | Descripción |
|---|-------|-------------|
| 2.1 | `MultiHeadAttention` | Atención con Q, K, V, softmax por cabeza, concatenación, proyección de salida |
| 2.2 | `FeedForward` | MLP: `Linear → SiLU → Linear` (como Llama) |
| 2.3 | `TransformerBlock` | `Attention + FFN + residual connections + layer norm` |
| 2.4 | `Transformer` | Embedding + `N` blocks + lm_head (proyección a vocabulario) |
| 2.5 | Reemplazar bigram | Usar `Transformer` en vez de `BigramModel` |
| 2.6 | Mini-entrenamiento (opcional) | Script Python para entrenar pesos chicos (< 1M params) |

**Resultado**: El servidor genera con un transformer real (con atención). ✅

---

## Fase 3 — KV-cache + scheduler

| # | Tarea | Descripción |
|---|-------|-------------|
| 3.1 | `KVCache` | Cachear keys/values por capa, por request |
| 3.2 | Attention con KV-cache | Modificar `MultiHeadAttention` para usar cache en vez de recomputar |
| 3.3 | Pool de slots | N slots de KV-cache para N requests concurrentes |
| 3.4 | Scheduler FCFS | Desencolar según orden de llegada cuando haya slot libre |
| 3.5 | Endpoint `/status` | Devuelve cola, slots ocupados, tiempo estimado |

---

## Fase 4 — Mejoras (futuro)

| # | Tarea | Descripción |
|---|-------|-------------|
| 4.1 | Parser GGUF / safetensors | Cargar pesos de modelos reales |
| 4.2 | Arquitectura real | Adaptar `Transformer` a Llama, Mistral, Qwen, etc |
| 4.3 | Batching | Procesar múltiples sequences en un forward pass |
| 4.4 | Sampling avanzado | top-k, top-p, repetition penalty, temperature |
| 4.5 | Streaming SSE | Devolver tokens al cliente a medida que se generan |
| 4.6 | Endpoint `/cancel` | Cancelar un request en curso |
| 4.7 | Archivo de configuración | Puerto, modelo, temperatura default, etc |

---

## Diagrama de flujo del servidor

```
POST /prompt
     │
     ▼
Parse JSON ──❌──> 400 {"error": "..."}
     │
     ▼
Tokenizer (texto → tokens)
     │
     ▼
InferenceRequest { prompt, tokens, temperature, ... }
     │
     ▼
ServerQueue.post(request) ──> asigna ID, encola
     │
     ▼
Responde {"id": 123, "status": "queued"}
     │
     ▼
[Worker thread] ──> desencola cuando hay slot libre
     │
     ▼
Model.generate(request) ──> forward pass → sample → next token
     │
     ▼
Responde al cliente (callback / poll / streaming)
```
