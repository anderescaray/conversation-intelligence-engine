# T06: conversaciones sintéticas de evaluación

## Estado y alcance

El objetivo de `golden-v1` es **150 conversaciones en inglés: 100 chat y 50 voz**, con
los 77 intents de Banking77. Hay 73 intents con dos ejemplos y cuatro con uno; cada intent
aparece al menos una vez en chat. No hay cobertura completa intent × canal.

Los artefactos en `data/synthetic/candidates/` son **candidatos**, no un golden set.
`golden-v1` solo se exporta cuando las 150 muestras cumplen las cuotas y tienen aprobación
humana vigente. El revisor es **Ander**, no el agente de código ni otro LLM. El piloto y la
revisión humana son puertas explícitas: generar el resto requiere el OK del usuario.

Esta entrega es solo evaluación. No implementa splits sintéticos train/dev ni entrenamiento;
T03 entrena con Banking77 real. Tampoco implementa el eval runner/CI gate de T11.

## Generación label-first

`scripts.synth.scenarios` construye todo el plan de 150 escenarios antes de invocar al LLM.
La semilla fija los intents, el sentimiento por turno del cliente, resolución, solicitud de
escalación, canal, longitud, persona, verbosidad, ruido e inventario de placeholders.

- Sentimiento: `negative`, `neutral`, `positive`; solo turnos del cliente, en orden.
  Cinco patrones con 30 escenarios cada uno, barajados por semilla. No hay etiqueta de
  sentimiento global derivada del mismo operador que se quiere evaluar.
- `resolved`: 75 verdaderos y 75 falsos, barajados independientemente del sentimiento.
- `escalation_requested`: solicitud explícita de supervisor/escalación, 30 casos. No indica
  que se haya realizado la derivación y **no es** `SummaryOutput.escalation_risk`.
- Entre tres y cinco turnos del cliente, alternados con el agente: 6–10 turnos totales.
- 30 controles sin PII; los demás incluyen dos tipos de PII, con hasta tres repeticiones
  de cada placeholder. Ambos canales cubren los seis tipos en el plan completo.
- Personas: cliente nuevo en banca móvil, experimentado u ocupado; verbosidad breve/media;
  ruido ligero o ausente. El ruido no se aplica después de insertar PII.

El único acceso generativo es `LLMClient.structured(draft_schema(escenario), prompt, ...)`,
con `load_prompt("synthetic", 1)`. El esquema fija la cantidad exacta de turnos del escenario,
no solo un rango, y los errores activan la reparación estructurada del cliente.
El modelo escribe turnos, no decide ni reemplaza las etiquetas. `prompts/synthetic/v1.yaml`
no contiene ejemplos de conversaciones. Las etiquetas son objetivos hasta la revisión humana.

### Comandos

Desde la raíz, con Ollama y `qwen3:8b` ya disponibles:

```bash
uv sync
uv run python -m scripts.synth.generate --n 10
```

Default: proveedor Ollama, modelo `qwen3:8b`, semilla 42, destino nuevo ignorado por Git
`data/generated/synthetic/run-<timestamp>`, timeout de 300 segundos por llamada lógica y hasta tres
intentos por escenario. El cliente LLM gestiona su propia reparación de esquema y reintentos
transitorios; su uso reportado se acumula en tokens/costes. La salida del CLI contiene la ruta
de salida, IDs, contabilidad y causas de rechazo, nunca el diálogo ni valores PII.

Para registrar también el digest conocido del modelo y escanear nombres con el modelo local:

```bash
uv sync --group ml
uv run --group ml python -m scripts.synth.generate --n 10 --seed 42 \
  --model-revision 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41 \
  --ner-model en_core_web_lg --output data/synthetic/candidates/pilot-v1.1
```

El digest anterior identifica el modelo instalado durante T06, no todos los modelos con ese
mismo tag. Verificarlo en Ollama antes de reutilizarlo. Si se omite `--model-revision`, queda
`unresolved`, no un hash inventado. `model_id` siempre está registrado.

`--resume` solo acepta la misma semilla, generador, prompt, modelo, lockfile y nivel de escaneo;
omite candidatos existentes y no reinicia el presupuesto de intentos agotados. Puede ampliarse
`--n` hasta 150 **solo después del OK al piloto**. No se sobrescriben otros directorios.
Usar un único proceso escritor por directorio; los JSONL se sincronizan tras cada registro.
Una línea corrupta hace fallar la lectura explícitamente, no se ignora ni se aprueba.

Anthropic es opcional: `--provider anthropic --model claude-haiku-4-5`, con la clave existente
en `Settings`. No hay fallback de pago ni SDK directo. No se modifican `config.py` o sus defaults.
`--trace` activa el setup existente de Phoenix. El CLI oculta entradas y salidas OpenInference
antes de instrumentar, incluso borradores inválidos. Se emiten spans `synthetic.*` y métricas
compartidas de latencia/errores/tokens/coste sin texto ni notas de revisión.

## Formato y procedencia

En cada directorio de candidatos:

| Archivo | Contenido |
| --- | --- |
| `run.json` | Identidad y configuración inmutable de la ejecución. |
| `candidates.jsonl` | Ledger autoritativo: `Sample(conversation, labels)`; preserva reemplazos. |
| `conversations.jsonl` | Vista de candidatos actuales; cada línea valida como `Conversation`. |
| `labels.jsonl` | `Annotation` enlazada por ID: escenario, PII, estado y procedencia. |
| `attempts.jsonl` | Intentos aceptados/rechazados automáticamente, latencias, tokens y causas. |
| `report.json` | Distribuciones reales, rechazos, costes y cuotas pendientes. |
| `review.jsonl` | Ledger humano append-only, creado únicamente al revisar. |

`accepted` en un intento significa **validación automática**, no aprobación humana. El
informe guarda `human_approved` en el momento de generarse; la fuente vigente de revisión
es `review.jsonl`. Las distribuciones del informe se calculan de los candidatos actuales.

Cada muestra contiene semilla global y de muestra, versión del generador/esquema, modelo y
revisión, prompt y hash de su representación validada, hash de `uv.lock`, temperatura,
intento, tokens, coste y latencia. IDs e inicio sintético no dependen del reloj.

La semilla reproduce escenarios y PII con las versiones fijadas. **No promete reproducir
bit a bit la respuesta del LLM**: el cliente actual no expone una semilla de inferencia.
Conservar los JSONL es la forma de reproducir exactamente los textos de evaluación.

## PII textual y hablada

El modelo solo recibe tokens como `[[PERSON_1]]`. Tras validar el borrador, código introduce:

| Tipo | Procedencia |
| --- | --- |
| `PERSON` | Faker `en_US`, instancia con semilla local; sin personas fuente. |
| `CREDIT_CARD` | PAN públicos de prueba de Stripe, válidos para Luhn; nunca usados en pagos. |
| `IBAN_CODE` | Ejemplos públicos de formato IBAN GB/DE y checksum válido; sin consultas bancarias. |
| `PHONE_NUMBER` | Rango ficticio NANP `+1 202 555 0100–0199`. |
| `EMAIL_ADDRESS` | Identificadores generados bajo el dominio reservado `example.com`. |
| `ACCOUNT_NUMBER` | Identificadores ficticios con prefijo 9900; sin vinculación a cuentas. |

Las fábricas no consultan registros de personas o cuentas. Un nombre o número ficticio puede
coincidir accidentalmente con uno existente: no se afirma que los checksums creen cuentas
sandbox ni que los IBAN de ejemplo sean utilizables. No contactar ni operar con estos valores.

Cada valor guarda `canonical_value`, `spoken_form` y `surface_form`. En chat la superficie
es canónica. En voz, números se expresan dígito a dígito; emails se deletrean con `at`, `dot`,
`dash` o `underscore`; IBANs combinan letras y dígitos hablados. Los nombres siguen como nombres.
El TTS lee esa misma superficie: no existe una transformación oculta posterior.

Cada aparición tiene tipo, turno, token y offsets `[start,end)` en **caracteres Python del texto
original de ese turno**, no bytes ni posiciones en `Conversation.text`. Incluye repeticiones.
`text[start:end]` debe ser exactamente la superficie almacenada; `score=1` expresa inserción
conocida, no confianza de un detector. Los offsets de voz también son exactos para su texto
fuente hablado, pero no se reutilizan sobre una transcripción ASR distinta.

### Evaluación después de ASR

El diseño de evaluación, que consumirá T11, distingue:

1. **PII sobre texto fuente:** comparar spans de T02 con los offsets exactos de referencia.
2. **PII end-to-end tras ASR:** extraer los valores de las entidades predichas usando los
   offsets del propio texto ASR, normalizarlos y compararlos con `canonical_value` por
   `(conversación, speaker, tipo, valor)`. Medir precisión/recall a nivel de valor único;
   no comparar offsets ni índices de turnos originales con los de Whisper. Reportar
   repeticiones/ocurrencias separadamente, sin inflar el recall con el mismo valor repetido.
3. Una PII perdida o irreconocible por ASR es un fallo end-to-end. Separar, si se diagnostica,
   error de reconocimiento y error del detector condicionado a que el valor sí esté presente.
4. Intent, resolución y solicitud de escalación mantienen la etiqueta humana original.
   Para trayectoria, alinear segmentos ASR con turnos fuente por speaker y solapamiento temporal
   usando el `Conversation` enriquecido por TTS; no asumir igual segmentación o número de turnos.
   No cambiar etiquetas para acomodar errores de ASR.

`pii.value_key` proporciona normalización básica de mayúsculas, puntuación, dígitos hablados,
`oh`, `double`/`triple` y emails deletreados. Recibe **un valor extraído**, no el transcript entero.
No resuelve todas las homofonías ni numerales compuestos: los casos ambiguos requieren revisión
alineada, no fuzzy matching silencioso. El redactor actual de T02 no reconoce números hablados;
es una limitación que el conjunto debe revelar, no un motivo para eliminar esos ejemplos.

## Validación y revisión humana

Automáticamente se comprueban contratos/esquemas, estructura de turnos, inventario exacto de
placeholders, offsets, PII sospechosa fuera de los spans insertados, evidencia léxica de
escalación y duplicados. Dedup usa texto con PII enmascarada, normalización Unicode y Jaccard
de 5-gramas de palabras con umbral 0.85. Se conserva la causa y contabilidad de rechazos,
no sus textos. Agotar intentos deja índices pendientes y devuelve código de salida no cero.

El escaneo base usa patrones conservadores; `--ner-model en_core_web_lg` añade detección local
de PERSON, sin descargas durante generación. Ninguno garantiza ausencia absoluta de PII.
La evidencia léxica de supervisor/escalación no demuestra su semántica: ambos límites hacen
obligatoria la revisión humana. No se usan predicciones de T02/T03 como ground truth ni se
exige recall perfecto de T02 para admitir una muestra.

### Protocolo de Ander

```bash
uv run python -m scripts.synth.review \
  --dataset data/synthetic/candidates/pilot-v1.1 --reviewer ander --batch-size 10
```

Para cada muestra, verificar:

1. El intent principal encaja en la taxonomía; no existe otra intención principal competidora.
2. Cada turno **del cliente** coincide con su sentimiento asignado; no confundir cortesía del agente.
3. Resolución y solicitud de escalación tienen evidencia explícita y no contradictoria.
4. Toda PII procede de las inserciones visibles, con tipo y límites correctos; no quedan datos
   personales inventados por el modelo, placeholders ni valores no etiquetados.
5. Naturalidad, diversidad, coherencia de repeticiones y, en voz, inteligibilidad de superficies.
   Escuchar el audio regenerado antes de aprobar una muestra de voz para el golden completo.

Acciones: `a` approve, `r` reject, `n` note, `s` skip, `q` quit. Reject/note requieren razón.
Una nota no es aprobación ni revoca una decisión anterior. Para revocar, usar reject con
`--revisit`. El CLI muestra notas previas y salta decisiones terminales al reanudar. Cada evento
registra ID, SHA-256 del candidato completo, revisor, acción, nota y fecha UTC. Una modificación
invalida las decisiones de la versión anterior. No incluir PII real en notas.

Los candidatos rechazados permanecen en el historial. Para sustituir explícitamente el índice
rechazado N, repetir el comando de generación original con `--resume --regenerate-index N`.
Solo admite índices rechazados por una persona; no reemplaza aprobados ni pendientes. Conserva
el escenario, asigna nuevos intentos y exige revisar de nuevo la nueva versión. Si se usa
`--review-jsonl` para una ruta alternativa, pasar la misma ruta a review, generación y exportación.

Tras generar y aprobar las 150 muestras:

```bash
uv run python -m scripts.synth.review \
  --dataset data/synthetic/candidates/pilot-v1.1 --reviewer ander \
  --export data/synthetic/golden-v1
```

El exportador comprueba cuotas, dedup y hashes vigentes. Solo entonces escribe conversaciones,
etiquetas `approved`, aprobaciones y manifiesto. No sobrescribe una versión existente. El hash
de revisión corresponde al registro original con `status=candidate`; el cambio de estado en
la vista publicada no representa una revisión diferente. Sin 150 aprobaciones, no existe golden-v1.

## Banking77 y leakage

**El generador no lee ni utiliza utterances de Banking77 de ningún split**: no hay ejemplos
de test ni train en prompts, few-shot, validadores generativos o instrucciones al modelo.
Solo usa los nombres de las 77 clases publicados en la metadata de
[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77/blob/796a4623935746f71378f0ebd435635a8ce08e50/README.md),
revisión `796a4623935746f71378f0ebd435635a8ce08e50`, CC BY 4.0. Se conservan sus nombres exactos,
incluso `Refund_not_showing_up` y `reverted_card_payment?`. Atribución: PolyAI, Casanueva et al.,
*Efficient Intent Detection with Dual Sentence Encoders*, 2020.

La auditoría es independiente y nunca se importa desde el generador:

```bash
uv run python -m scripts.synth.audit \
  --conversations data/synthetic/candidates/pilot-v1.1/conversations.jsonl \
  --banking77-test-csv /ruta/al/test.csv --output data/generated/holdout-audit.json
```

No descarga datos: recibe el snapshot local de test usado por T03. Compara únicamente turnos
del cliente, normalizando Unicode, mayúsculas, espacios y puntuación. Guarda hash del test,
número de filas y referencias a coincidencias, nunca sus textos. No certifica similitud semántica
ni descarta memorización en el preentrenamiento del LLM.

La fixture offline `tests/synth/banking77_test_fixture.jsonl` contiene las tres primeras
utterances de test de `PolyAI-LDN/task-specific-datasets`, revisión
`9d081458ff52e53cf7e848f414e6e9344e4e6696`, archivo `banking_data/test.csv`, CC BY 4.0.
Solo la usan los tests de auditoría. Estos comprueban tanto detección de una copia deliberada
como ausencia de coincidencias en los JSONL generados versionados. Tres filas no sustituyen
la auditoría del snapshot completo de T03.

## Audio regenerable, versiones y licencias

No se versionan WAV, modelos, LFS ni DVC. `data/generated/` ya está ignorado por Git.
Primero, descargar explícitamente los dos modelos fijados; la carga de inferencia nunca descarga:

```bash
uv run python -m scripts.synth.audio --download-voices --voices-dir data/generated/voices
uv sync --group ml
uv run --group ml python -m scripts.synth.audio \
  --dataset data/synthetic/candidates/pilot-v1.1/conversations.jsonl \
  --voices-dir data/generated/voices --output data/generated/synthetic/pilot-v1
```

La segunda orden puede consumir también el golden publicado. No llama al LLM. Produce WAV
PCM16/16 kHz con cliente a la izquierda, agente a la derecha y 250 ms de silencio entre turnos.
El otro canal permanece silencioso. Emite `conversations.jsonl` con `audio_uri` local y timestamps
medidos en frames, directamente utilizable por `WhisperTranscriber` de T01. No inventa confianza ASR.

`manifest.json` registra versión de Piper, revisión/IDs y hashes de modelos y configuraciones,
speaker IDs, parámetros de síntesis, SHA-256 por WAV, duración, tiempo de render y hash del JSONL
fuente/enriquecido. Dos modelos distintos proporcionan voces distintas. `noise_scale=0` y
`noise_w_scale=0` reducen aleatoriedad; diferencias de plataformas o runtime pueden alterar bytes.
Regenerar crea un directorio nuevo y su manifiesto, sin sobrescribir una referencia anterior.

| Rol | Modelo / voz | Dataset y licencia |
| --- | --- | --- |
| Cliente | `en_US-ljspeech-high`, voz única | [LJ Speech](https://keithito.com/LJ-Speech-Dataset/), dominio público. |
| Agente | `en_US-libritts-high`, speaker 0 / `p3922` | [LibriTTS](https://www.openslr.org/60/), CC BY 4.0. Atribución: Heiga Zen et al., *LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech*, 2019, Google. |

Ambos modelos fueron entrenados desde cero según sus model cards; no se escogieron voces
Lessac ni Ryan con condiciones de investigación o no comerciales. Modelos publicados por
`rhasspy/piper-voices`, cuya metadata declara MIT; revisión fijada
`c10ece1aade47bb51c153c893d14e5bf8e5b7117`. Se conserva atribución a autores y datasets.
Model cards y SHA-256 completos están en `scripts/synth/voices.py` y en cada manifiesto.

El **motor Piper 1.3.0** usa GPL-3.0-or-later; no confundir la licencia del motor con la de
voces/datasets. No se redistribuye el motor ni sus pesos en este repo. `piper-tts==1.3.0` se
incorpora al grupo opcional `ml`, con carga lazy y ejecución CPU. Faker `37.8.0` (MIT) pertenece
a `dev` como herramienta de preparación/pruebas. Se fija también `tzdata==2025.2` en `dev`
porque Faker lo requiere y el resolver había seleccionado una versión publicada el día anterior;
se prefiere una release verificada de 2025-03-23. No se usan fechas ni zonas horarias de Faker.
No se añadieron dependencias al servicio base, SDKs de proveedor ni bibliotecas de audio:
se reutilizan PyAV/numpy y `wave`.

## Piloto ejecutado (2026-10-04)

Equipo: WSL2/Linux, RTX 5060 Laptop de 8151 MiB; Ollama `qwen3:8b` Q4_K_M, digest registrado
por muestra; NER `en_core_web_lg` 3.8.0. **10 candidatos, 0 aprobaciones humanas** en
`data/synthetic/candidates/pilot-v1.1/`; no se ha generado el resto ni publicado `golden-v1`.

El preflight `pilot-v1/`, generador 1.0.0, se abortó tras ocho intentos completados rechazados
(siete por longitud, uno por escalación). Conserva su ledger y una nota de interrupción;
el consumo/latencia del intento en vuelo no se conoce. Una regresión offline reproduce el
problema: el esquema permitía 6–10 turnos aunque el escenario exigiese diez. El generador
1.0.1 fija la longitud exacta en el esquema estructurado; los mismos diez escenarios se
repitieron en un directorio nuevo, sin cambiar etiquetas ni ocultar el preflight.

| Métrica del piloto 1.0.1 | Valor |
| --- | --- |
| Canales / intents | 7 chat, 3 voz; 10 intents distintos |
| Intentos / rechazos automáticos | 15 / 5 (33.33 %) |
| Causas de rechazo | 1 error LLM por timeout, 1 escalación, 1 inventario PII, 2 orden de roles |
| Tiempo acumulado de intentos | 580.276 s (incluye un timeout de 300 s) |
| Media por candidato, incluidos rechazos | 58.028 s |
| Media del intento aceptado solamente | 22.556 s |
| Tokens reportados, entrada / salida | 7,995 / 13,226 |
| Coste de API | 0 USD, Ollama local |
| Resueltos / no resueltos | 3 / 7 |
| Solicitudes de escalación | 2 |
| Sentimiento por turno del cliente | 26 positivos, 5 neutrales, 11 negativos |
| Spans PII | 24: tarjeta 5, IBAN 3, teléfono 3, email 6, cuenta 3, persona 4 |

Los tokens del timeout no fueron reportados por el proveedor: cero en ese intento significa
**uso no recibido**, no ausencia de computación. Los contadores son consumo reportado, no una
medida exacta del trabajo interrumpido. Para proveedores de pago, un fallo LLM deja el coste
como desconocido (`null`) en lugar de asumir gratuidad. La medición de generación excluye
carga inicial de NER, descarga de modelos y TTS; no es una garantía de rendimiento futuro.

Los tres WAV se renderizaron realmente con Piper 1.3.0 en CPU: 109.0865 s de audio en
15.903 s de render por turnos, excluyendo carga de modelos. El manifiesto de referencia está
versionado como `audio-manifest.json`; los WAV permanecen en `data/generated/synthetic/pilot-v1.1/`
y están ignorados por Git. No se ejecutó inferencia ASR real en este piloto.

El piloto precede al pin preventivo de tzdata: usó 2026.5 y conserva honestamente el hash del
lock de aquella ejecución. El entorno final usa 2025.2; los tests regeneran toda su PII con
las semillas guardadas y comprueban igualdad exacta. No se han reescrito las huellas históricas.
Un nuevo experimento requiere un directorio nuevo; `--resume` rechaza un lock distinto.

**La validación automática no certifica coherencia semántica.** La inspección preliminar,
sin registrar decisiones en nombre de Ander, encontró:

- `000`: envío de dinero, aunque el escenario pide `exchange_via_app`; varios turnos no
  expresan claramente el sentimiento positivo asignado.
- `001`: repite un bloque y emplea un PAN como tratamiento del cliente; la trayectoria
  tampoco coincide claramente con la posición de la frustración.
- `002`: `resolved=true`, pero el agente solo promete cerrar la cuenta próximamente;
  varios turnos parecen neutrales, no positivos.

Estas observaciones no son aprobaciones/rechazos humanos. Los datos siguen siendo candidatos.
Se recomienda revisar y ajustar prompt/modelo antes de escalar; no retocar sus etiquetas para
hacerlos pasar. El checkpoint del piloto y su manifiesto deben conservarse al crear una nueva
versión experimental. Queda pendiente el OK del usuario para cualquier ampliación.

## Limitaciones y verificación

- Solo inglés y dominio bancario; textos y voces artificiales, no tráfico real.
- Una o dos muestras por intent no soportan estimaciones fiables de F1 por clase.
- Cuotas artificiales, estilos/personas limitados y correlaciones del diseño; no sirven para
  inferir prevalencias, riesgo calibrado ni métricas de negocio.
- TTS neutro no reproduce necesariamente la prosodia emocional indicada por etiquetas textuales.
- ASR puede perder PII hablada o cambiar segmentación; reportar esa pérdida por separado.
- Escaneo automático y revisión de una persona pueden fallar; no se afirma anotación infalible.
- El conjunto no debe usarse para entrenar ni calibrar T03/T02. Train/dev sintéticos son follow-up.

Pruebas: fake models de `convintel.llm.testing`, sockets bloqueados, PCM ficticio para audio,
exportadores OTel en memoria; sin Ollama, Anthropic, descargas ni generación real de corpus.
La ejecución real de generación/audio solo ocurre mediante los CLI anteriores, nunca en pytest.

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv lock --check
```

La decisión de audio regenerable sin DVC/LFS y revisión ligada al hash está propuesta para
ADR-0007. El humano la incorporará en otra PR de docs; T06 no crea ni modifica ADR-0006.
