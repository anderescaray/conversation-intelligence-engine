# T06: conversaciones sintéticas de evaluación

## Estado y alcance

El objetivo de `golden-v1` es **150 conversaciones en inglés: 100 chat y 50 voz**, con
los 77 intents de Banking77. Hay 73 intents con dos ejemplos y cuatro con uno; cada intent
aparece al menos una vez en chat. No hay cobertura completa intent × canal.

Los artefactos en `data/synthetic/candidates/` son **candidatos**, no un golden set.
`golden-v1` solo se exporta cuando las 150 muestras cumplen las cuotas, superan el juez y tienen
aprobación humana vigente. El revisor es **Ander**, no el agente de código ni otro LLM. El piloto y la
revisión humana son puertas explícitas: generar el resto requiere el OK del usuario.

Esta entrega es solo evaluación. No implementa splits sintéticos train/dev ni entrenamiento;
T03 entrena con Banking77 real. Tampoco implementa el eval runner/CI gate de T11.

## Generación label-first

El generador **3.0.0 / esquema 3** utiliza `prompts/synthetic/v3.yaml` (salida estructurada
por turno contra un esqueleto construido por código) y el juez ciego `prompts/synthetic_judge/v2.yaml`.
Los prompts `synthetic/v1`, `synthetic/v2`, `synthetic_judge/v1` y los pilotos v1, v1.1 y v2 son
históricos e inmutables. El lector conserva sus campos y hashes originales, pero no son elegibles
para revisión ni golden. No hay conversión implícita de etiquetas.

**IDs de escenario.** Las muestras v3 conservan IDs `synth-v2-<seed>-<index>`. El prefijo identifica
el **plan de escenarios v2** (reglas de muestreo de abajo), que no cambia en v3, y no la versión del
generador; esta queda en `run.json` y en `provenance.generator_version`/`schema_version`. Mantenerlos
permite comparar pilotos y generadores sobre exactamente los mismos escenarios.

`scripts.synth.scenarios` construye todo el plan de 150 escenarios antes de invocar al LLM.
Cada intent tiene una definición breve, escrita en código en `INTENT_DESCRIPTIONS`, almacenada
con la muestra y enviada al generador y al juez. Son definiciones operativas revisables, no
utterances ni definiciones supuestamente oficiales de Banking77.

### Reglas de muestreo v2

- Sentimiento: `start_sentiment` y `end_sentiment` del **primer y último turno del cliente**.
  Valores: `negative`, `neutral`, `positive`. No se asignan etiquetas a turnos intermedios.
  No se modifica el contrato `SentimentOutput` de T02; estas son etiquetas del dataset.
- `resolved`: 75 verdaderos y 75 falsos, barajados por semilla; las emociones se condicionan
  a la resolución y escalación, en lugar de muestrearse de forma independiente.
- `escalation_requested`: 30 solicitudes explícitas del cliente, no riesgo ni derivación
  realizada. Todas empiezan negativas. Si siguen sin resolver, terminan negativas.
- Sin escalación, el inicio se muestrea con pesos negativo:neutral:positivo = 5:4:1.
- Con resolución, el final se muestrea con pesos positivo:neutral = 4:1.
- Sin resolución ni escalación, el final usa pesos negativo:neutral = 7:3, excepto **tres
  casos de los 75 no resueltos (4 %)**: empiezan neutrales y terminan positivos por la ayuda
  recibida, sin afirmar que el problema esté resuelto. Nunca pertenecen a escalaciones.
- Tres o cuatro turnos del cliente (6 u 8 totales), alternados con el agente. El esquema
  exige la cantidad exacta; el validador además impone límites de 6–10 turnos.
- 30 controles sin PII; los demás tienen dos valores completos. Cada token tiene turno y
  speaker fijados por código y debe aparecer **una sola vez** en esa posición.
- Toda PII completa pertenece al cliente, en los turnos 0 y 2. En 30 escenarios se añade una
  confirmación parcial del agente en el turno siguiente al valor original: solo sus últimos
  cuatro dígitos, calculados por código y unidos al origen mediante `source_token`.
- Personas: cliente nuevo en banca móvil, experimentado u ocupado; verbosidad breve/media;
  ruido ligero o ausente. El ruido no se aplica después de insertar PII.

Todo acceso LLM pasa por `LLMClient.structured()`. El modelo escribe turnos, no decide ni
reemplaza etiquetas. Ninguno de los prompts contiene utterances de Banking77. Un fallo del
juez provoca rechazo/regeneración con el **mismo escenario**, nunca ajuste de etiquetas.

### Esqueleto por turno (v3)

`schema.skeleton(scenario)` fija por código, para cada turno, `turn_index`, `speaker`, los
placeholders exactos que debe contener (posiblemente ninguno) y pistas breves: tipo de PII,
confirmación de últimos cuatro dígitos y sentimiento del primer/último turno del cliente. El modelo
devuelve una lista `{turn_index, speaker, text}` de longitud exacta. `validation.skeleton_errors`
comprueba longitud, índice, speaker y que cada turno contenga **exactamente** sus placeholders, una
vez cada uno, y ningún otro. Ante un desajuste se hace **una** llamada de reparación: el código
envía los errores concretos y la respuesta previa como datos en un bloque `<repair>`; cómo usarlo
lo dice el prompt versionado. Si sigue fallando: `skeleton_mismatch`. Las demás causas
(repetición, PII ajena, escalación léxica, juez) se rechazan sin reparación. La reparación consume
el mismo deadline de la muestra.

### Comandos

Desde la raíz, con Ollama y `qwen3:8b` ya disponibles:

```bash
uv sync
uv run python -m scripts.synth.generate --n 10
```

Default: proveedor Ollama, modelo `qwen3:8b`, semilla 42, destino nuevo ignorado por Git
`data/generated/synthetic/run-<timestamp>` y hasta tres intentos por escenario dentro de un
**presupuesto total de 120 s por muestra**: generación, reparaciones/reintentos del cliente,
validación, inserción de PII y juez comparten el mismo deadline. No se reinicia por intento.
Al expirar se registra `sample_timeout`, se omite esa muestra y se continúa con la siguiente.
`--timeout` permite reducirlo, no superar 120. Reanudar no vuelve a intentar muestras expiradas.
La carga inicial de modelos/NER y la persistencia del ledger no forman parte de ese presupuesto.

Se desactiva `reasoning` en Ollama tanto para generación como para juez, a través de modelos
creados por `convintel.llm.build_chat_model` e inyectados en `LLMClient`. El adaptador lo envía
como `think=false`; no hay SDK directo ni cambios de `Settings`. Generación usa temperatura
0.35 y juez 0, en clientes separados. El juez es independiente del generador:
`--judge-provider/--judge-model/--judge-model-revision` (por defecto, el mismo que el generador).
`run.json` y `report.json` registran `model_id` y `judge_model_id`.
La salida del CLI contiene IDs, contabilidad, causas y veredictos tipados, nunca diálogos ni PII.

Para registrar también el digest conocido del modelo y escanear nombres con el modelo local:

```bash
uv sync --group ml
uv run --group ml python -m scripts.synth.generate --n 10 --seed 42 \
  --model-revision 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41 \
  --judge-provider anthropic --judge-model claude-haiku-4-5 \
  --ner-model en_core_web_lg --output data/synthetic/candidates/pilot-v3-qwen
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

`accepted` significa **validadores y juez superados**, nunca aprobación humana.
El informe guarda `human_approved` al generarse; la fuente vigente de revisión es `review.jsonl`.
Las distribuciones se calculan solo sobre candidatos admitidos. Los rechazos no se publican
como muestras: conservan su causa, fase, contabilidad y, si respondió el juez, su veredicto.

Cada muestra contiene semilla global y de muestra, definición del intent, posiciones de PII,
versión del generador/esquema, modelo y revisión, prompt/hash, hash de `uv.lock`, temperatura,
razonamiento configurado, intento, tokens, coste y latencia. IDs e inicio sintético no dependen
del reloj. `labels.judge` guarda modelo/revisión, prompt/hash, temperatura, `reasoning`, uso,
latencia, hash de la entrada enmascarada, las cinco observaciones con índices de evidencia,
`mismatched_labels` y `evidence_errors`. `provenance.repaired` indica si hubo reparación.

`report.json` distingue categorías excluyentes `validator`, `judge`, `timeout`, `llm` y sus
tasas sobre todos los intentos. Las causas detalladas y `judge_failed_labels` permiten separar
fallos de intent, resolución, escalación y sentimiento; estos últimos pueden solaparse en
un mismo rechazo. `sample_elapsed_s` suma los intentos de cada ID, incluidos rechazos, y
`timed_out_ids` identifica omisiones. Los contadores de tokens incluyen ambas llamadas y
reparaciones reportadas; un timeout puede tener uso no reportado.

Cada intento guarda `generation_usage` y `judge_usage` (segundos, tokens y coste de esa fase,
reparaciones incluidas). `report.json` los agrega en `phase_totals`, `phase_mean_per_scenario`
(sobre escenarios intentados) y `phase_per_admitted` (coste total dividido por admitidos), además
de `repairs` y `repairs_fixed`. Si una llamada de pago se interrumpe, su coste es `null`.

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
original de ese turno**, no bytes ni posiciones en `Conversation.text`. En v2 cada token tiene
una única aparición y ubicación; las repeticiones de v1 se conservan solo como datos históricos.
`text[start:end]` debe ser exactamente la superficie almacenada; `score=1` expresa inserción
conocida, no confianza de un detector. Los offsets de voz también son exactos para su texto
fuente hablado, pero no se reutilizan sobre una transcripción ASR distinta.

Una `Placement` fija `turn_index`, `speaker`, `exposure` y `source_token`. Los valores
`last_four` tienen como canónico solo ese sufijo, nunca se inventan por separado. No se
permiten valores completos en voz del agente, ni siquiera fuera de placeholders: se buscan
también sus formas normalizadas. Las confirmaciones parciales se anotan con offsets exactos,
pero **no cuentan como una tarjeta/IBAN/teléfono/cuenta completa** para recall. Filtrar los
valores `exposure=full` para esa métrica; medir confirmaciones parciales por separado.

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
   Para sentimiento inicio/fin, agrupar segmentos ASR por speaker y solapamiento temporal con
   el primer y último turno fuente del cliente usando el `Conversation` enriquecido por TTS.
   No equiparar automáticamente el último segmento ASR con todo el último turno del cliente.
   No se evalúa una trayectoria intermedia ni se cambian etiquetas para acomodar errores ASR.

`pii.value_key` proporciona normalización básica de mayúsculas, puntuación, dígitos hablados,
`oh`, `double`/`triple` y emails deletreados. Recibe **un valor extraído**, no el transcript entero.
No resuelve todas las homofonías ni numerales compuestos: los casos ambiguos requieren revisión
alineada, no fuzzy matching silencioso. El redactor actual de T02 no reconoce números hablados;
es una limitación que el conjunto debe revelar, no un motivo para eliminar esos ejemplos.

## Validación y revisión humana

Antes del juez se comprueban contratos/esquemas, límites y orden de turnos, posición y speaker
exactos de cada placeholder, offsets, valores deterministas y PII ajena a las inserciones.
Se rechaza PII completa del agente y la aparición literal de la etiqueta técnica del intent.
Se rechazan turnos repetidos y bloques repetidos tras normalizar Unicode, mayúsculas,
puntuación y enmascarar PII. El dedup entre muestras usa además Jaccard de 5-gramas (0.85).
Agotar intentos o el tiempo deja índices pendientes y un código de salida no cero.

Después se llama al **juez ciego** (`synthetic_judge/v2`), con contexto nuevo. Solo recibe el
diálogo enmascarado y la taxonomía de `INTENT_DESCRIPTIONS`; **nunca las etiquetas objetivo**.
No decide si algo "coincide": extrae lo que observa (intent, `resolved`, `escalation_requested`,
sentimiento del primer y último turno del cliente), cada valor con índices de evidencia. El intent
está restringido en el esquema a los 77 nombres. El código compara después observación y objetivo;
cualquier diferencia es `label_mismatch`, sin mayorías. El juez v1, que devolvía `matches` por
objetivo, leía los objetivos `false` como "no observado" y rechazaba muestras válidas.

El código valida la evidencia (`invalid_evidence` si falla): no vacía en los cinco campos e
índices existentes; sentimiento y escalación citan **solo turnos del cliente**, y el sentimiento
inicial/final debe incluir el primer/último turno del cliente. Para `escalation_requested=false`
el juez cita los turnos del cliente revisados, porque una ausencia no tiene un único turno.
Resolución futura/prometida no cuenta como completada. Sin explicaciones libres ni texto del
diálogo en la salida del juez.

Solo se publican candidatos con el juez superado. La observación queda unida por SHA-256 a la
entrada enmascarada y a la taxonomía, y la revisión humana al hash de la muestra completa;
modificar cualquiera vuelve la muestra inelegible. `reviewable()` recompara además observación y
escenario. Review/export filtran candidatos sin juez, rechazados, desactualizados o históricos.
Superar el juez **no sustituye tu aprobación**.

### Puerta de controles del juez

Antes de usar un juez en un piloto, `scripts.synth.judge_check` lo ejecuta sobre
`tests/synth/judge_controls.jsonl`: diez diálogos escritos a mano con `truth` (lo que el texto
muestra) y `targets`. Tres cumplen todas las etiquetas; siete rompen **exactamente un** campo
(intent vecino, resolución solo prometida, resolución completada marcada como abierta, escalación
solo ofrecida por el agente, petición de manager con objetivo `false`, inicio neutral con objetivo
negativo y un «ok, thanks» seco con objetivo positivo). Un juez pasa solo si no marca nada en los
válidos y marca exactamente el campo roto en los demás, con evidencia válida. Es una **puerta,
no una estimación de exactitud**: los resultados se dan como «x/10 controles».

```bash
uv run python -m scripts.synth.judge_check --provider anthropic --model claude-haiku-4-5 \
  --output data/synthetic/judge-checks/claude-haiku-4-5.json
```

Resultados (2026-10-18), `data/synthetic/judge-checks/`:

| Campo: observado = verdad | qwen3:8b | claude-haiku-4-5 |
| --- | ---: | ---: |
| intent | 10/10 | 10/10 |
| resolved | 9/10 | 10/10 |
| escalation_requested | 10/10 | 10/10 |
| start_sentiment | 10/10 | 10/10 |
| end_sentiment | 9/10 | 10/10 |
| **Controles superados** | **2/10** | **10/10** |
| Tiempo / coste | 42.9 s / 0 USD | 21.1 s / 0.049 USD |

qwen3:8b observa casi siempre el valor correcto, pero en 8 controles cita turnos del agente como
evidencia de escalación ausente (y en uno, de sentimiento final), y en dos marca además un campo no
roto. No pasa la puerta. **Haiku es el juez de ambos pilotos v3**, de modo que estos comparan solo
generadores.

El escaneo base usa patrones conservadores; `--ner-model en_core_web_lg` añade detección local
de PERSON, sin descargas durante generación. Ninguno garantiza ausencia absoluta de PII.
La evidencia léxica de supervisor/escalación no demuestra su semántica: ambos límites hacen
obligatoria la revisión humana. No se usan predicciones de T02/T03 como ground truth ni se
exige recall perfecto de T02 para admitir una muestra.

### Protocolo de Ander

```bash
uv run python -m scripts.synth.review \
  --dataset data/synthetic/candidates/pilot-v3-haiku --reviewer ander --batch-size 10
```

Para cada muestra, verificar:

1. El intent principal encaja en la taxonomía; no existe otra intención principal competidora.
2. El primer y último turno **del cliente** coinciden con inicio/fin; no confundir cortesía
   del agente con emoción del cliente. Comprobar también la observación y evidencia del juez.
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
  --dataset data/synthetic/candidates/pilot-v3-haiku --reviewer ander \
  --export data/synthetic/golden-v1
```

El exportador exige juez aprobado/vigente, aprobación humana, cuotas, dedup y hashes. Entonces escribe conversaciones,
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
  --conversations data/synthetic/candidates/pilot-v3-haiku/conversations.jsonl \
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
  --dataset data/synthetic/candidates/pilot-v3-haiku/conversations.jsonl \
  --voices-dir data/generated/voices --output data/generated/synthetic/pilot-v3-haiku
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

## Pilotos v3 (2026-10-18)

Mismos diez escenarios de la semilla 42, generador 3.0.0, NER local, deadline de 120 s por
muestra y **el mismo juez para ambos**: `claude-haiku-4-5` con `synthetic_judge/v2`. Solo cambia
el generador. Artefactos en `data/synthetic/candidates/pilot-v3-{haiku,qwen}/`. **0 aprobaciones
humanas**: son candidatos pendientes de revisión.

| Métrica | Haiku genera | qwen3:8b genera |
| --- | ---: | ---: |
| Admitidos (validadores + juez) | **8/10** | 4/10 |
| Admitidos en el primer intento | 5/10 | 1/10 |
| Intentos / rechazos | 17 / 9 | 26 / 22 |
| Rechazos del juez (`label_mismatch`) | 8 | 14 |
| Rechazos de validadores | 1 `escalation_evidence` | 6 `escalation_evidence`, 2 `skeleton_mismatch` |
| Campos fallidos según el juez | inicio 4, final 3, resolved 2 | resolved 7, intent 5, inicio 3, final 2 |
| Reparaciones / corregidas | 0 / 0 | 2 / 0 |
| Pendientes | 007, 008 | 000, 001, 004, 005, 006, 008 |
| Generación por escenario (media) | 6.8 s · 0.0066 USD | 15.6 s · 0 USD (local) |
| Juez por escenario (media) | 3.8 s · 0.0087 USD | 4.4 s · 0.0096 USD |
| Generación por admitido | 8.4 s · 0.0082 USD | 38.9 s · 0 USD |
| Juez por admitido | 4.8 s · 0.0108 USD | 11.1 s · 0.0240 USD |
| Coste total del piloto | 0.152 USD | 0.096 USD |

El esqueleto elimina los fallos de PII que bloquearon v2: ningún rechazo por
`pii_placement`/`placeholder_inventory` en 43 intentos. Los rechazos restantes son semánticos y
los detecta el juez; no hubo timeouts ni errores LLM. qwen3:8b no consigue escribir la petición
explícita de escalación (6/6 rechazos léxicos en 001 y 006). En ambos generadores, el escenario 008
(inicio negativo en una pregunta informativa de `visa_or_mastercard`) falla siempre en el
sentimiento inicial, y Haiku tiende a cerrar con un final positivo cuando el objetivo es neutral (007).

Observaciones preliminares sobre candidatos de Haiku, **no decisiones de revisión**:
`001` resulta forzado porque el plan de PII exige un IBAN en un top-up con tarjeta; en `003` el
agente ofrece derivar el caso al equipo de pagos, aunque el prompt pide no ofrecer escalación
(la etiqueta del cliente sigue siendo correcta); en `002` el final positivo es moderado.
No se ha generado nada más allá de estos diez escenarios ni se ha renderizado audio.

## Piloto v2: resultado bloqueante (2026-10-04)

Se ejecutaron exactamente los diez escenarios iniciales de la semilla 42 con Qwen3:8b
Q4_K_M, `reasoning=false`, NER local y deadline de 120 s por muestra. **Ninguno superó
el proceso completo**: no hay candidatos que mostrar como aprobados por el juez ni
muestras disponibles para revisión humana. No se generaron casos adicionales ni se
relajaron criterios para completar la cuota. No se renderizó audio nuevo.

Artefactos: `data/synthetic/candidates/pilot-v2/{run.json,attempts.jsonl,report.json}`;
las vistas `conversations.jsonl` y `labels.jsonl` están vacías deliberadamente. No existe
ledger de candidatos aceptados ni revisión humana de este piloto.

| Métrica | Resultado |
| --- | --- |
| Escenarios / intentos | 10 / 30 (tres por escenario) |
| Admitidos por validadores y juez | 0 |
| Rechazados por validadores | 28/30 = 93.33 % |
| Rechazados por juez | 2/30 = 6.67 %; 2/2 de los que llegaron al juez |
| Rechazo total | 100 % |
| Timeout / errores LLM | 0 / 0 |
| Tiempo acumulado de intentos | 119.889 s |
| Media por intento / por escenario | 3.996 s / 11.989 s |
| Tokens reportados entrada / salida | 25,933 / 5,190 |
| Coste de API | 0 USD |

Motivos: ubicación de PII 13, inventario de placeholders 11, bloques repetidos 2,
turno repetido 1, PII no insertada por código 1, etiquetas incompatibles según el juez 1,
y evidencia inválida del juez 1. Ambos veredictos marcaron incompatibilidad en resolución
y solicitud de escalación; en uno, la evidencia del sentimiento final señaló al agente.
Los registros guardan modelo, prompt, veredictos y hashes, no el texto rechazado.

| Índice | Canal | Segundos acumulados |
| --- | --- | ---: |
| 000 | chat | 23.132 |
| 001 | chat, escalación | 11.202 |
| 002 | voz | 10.618 |
| 003 | chat | 10.687 |
| 004 | chat | 8.324 |
| 005 | voz | 15.103 |
| 006 | chat, escalación | 9.880 |
| 007 | chat | 10.967 |
| 008 | voz | 11.266 |
| 009 | chat | 8.711 |

Los 119.889 s son la suma del piloto, **no un timeout global**; cada escenario quedó muy
por debajo de su límite. Las distribuciones de aceptados están vacías, no se rellenan con
las etiquetas planificadas. V2 y el prompt del juez v1 ya fueron utilizados y permanecen
inmutables a partir de este ensayo.

Queda bloqueada la petición de mostrar tres muestras que hayan pasado el juez. Antes de
otro piloto se propone fijar la estructura de cada turno también en la salida estructurada
(o ensamblarla por código) y comprobar la interpretación de objetivos booleanos del juez
con casos controlados. No se puede atribuir este resultado solo a `reasoning=false` sin
un experimento comparativo. Cualquier cambio de prompt requerirá nuevas versiones y el
siguiente ensayo necesita el OK del usuario; no se cambian etiquetas para aceptar textos.

## Piloto histórico v1 (2026-10-04)

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
- Escaneo automático, juez y revisión humana pueden fallar; no se afirma anotación infalible.
- En `pilot-v3-haiku`, Haiku juzga texto generado por Haiku: hay riesgo de **autopreferencia**
  (favorecer su propio estilo o compartir sus errores), así que su tasa de admisión puede estar
  inflada frente a `pilot-v3-qwen`, que usa el mismo juez sobre otro generador. La puerta de
  controles no mide ese sesgo; solo la revisión humana lo acota.
- En pilotos anteriores, generador y juez usaban Qwen3:8b en contextos distintos. Sus
  acuerdos no son una medición independiente de precisión ni una aprobación de ground truth.
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
