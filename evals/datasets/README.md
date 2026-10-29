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

El generador **3.4.0 / esquema 3** usa `prompts/synthetic/v5.yaml` y `prompts/synthetic_judge/v4.yaml`, que
comparten la guía de etiquetado v2 (sección «Generador 3.4»); el juez es `claude-sonnet-5-5`, que pasa la
puerta v3 (26/26 dos veces), y el generador sigue siendo Haiku.
El generador **3.3.0 / esquema 3** usaba el plan de escenarios v4 con la taxonomía v2, `prompts/synthetic/v4.yaml`
y `prompts/synthetic_judge/v3.yaml`, que comparten la guía de etiquetado v1 (sección «Generador 3.3»).
El generador **3.2.0** usaba el **plan de escenarios v4** (abajo), `prompts/synthetic/v3.yaml`
(salida estructurada por turno contra un esqueleto construido por código) y el juez ciego
`prompts/synthetic_judge/v2.yaml`. Ni 3.1 ni 3.2 cambian prompts, así que la puerta de controles
del juez sigue vigente. El generador 3.1.0 usaba el plan v3 y el 3.0.0, el plan v2.
Los prompts `synthetic/v1`, `synthetic/v2`, `synthetic_judge/v1` y los pilotos v1, v1.1 y v2 son
históricos e inmutables. El lector conserva sus campos y hashes originales, pero no son elegibles
para revisión ni golden. No hay conversión implícita de etiquetas.

**IDs de escenario.** El prefijo identifica el **plan de escenarios**, no la versión del generador
(que queda en `run.json` y en `provenance`). Los pilotos del generador 3.0.0 usan el plan v2 y
conservan `synth-v2-<seed>-<index>`, para compararse sobre los mismos escenarios. El plan v3 cambia
etiquetas y PII, así que sus muestras usan `synth-v3-<seed>-<index>`: un mismo índice no es el mismo
escenario en ambos planes. El plan v4 (generador 3.2) usa `synth-v4-<seed>-<index>`.
`scenarios.legacy_scenarios(seed)` reproduce el plan v2 y `scenarios(seed, plan=3)` el plan v3; los
tests comprueban que coinciden con los candidatos guardados de cada piloto.

`scripts.synth.scenarios` construye todo el plan de 150 escenarios antes de invocar al LLM.
Cada intent tiene una definición breve, escrita en código en `INTENT_DESCRIPTIONS`, almacenada
con la muestra y enviada al generador y al juez. Son definiciones operativas revisables, no
utterances ni definiciones supuestamente oficiales de Banking77.

### Planes de escenarios v3 (generador 3.1) y v4 (generador 3.2): plausibilidad por intent

`scripts/synth/plausibility.py` clasifica explícitamente los 77 intents (un test exige cobertura
exacta) en **tipo** y **familia de PII**. Solo se muestrea dentro de estas reglas, de modo que el
generador nunca recibe un objetivo contradictorio; `ScenarioV3` las vuelve a validar.

| Tipo (intents) | Escalación | Inicio | Final si resuelto | Final si no resuelto |
| --- | --- | --- | --- | --- |
| info (22): preguntas de producto o política | nunca | v3: neutral:positivo 4:1 · **v4: neutral** | v3: positivo:neutral 1:1 · **v4: positivo** | v3: neutral:negativo 3:2 · **v4: no existe** |
| problem (37): algo ha fallado | sí | negativo:neutral 3:1 | positivo:neutral 4:1 | negativo:neutral 7:3 |
| request (18): el cliente pide una acción | sí | neutral:negativo:positivo 6:3:1 | positivo:neutral 3:1 | negativo:neutral 1:1 |

Las escalaciones empiezan siempre negativas y, si no se resuelven, terminan negativas. Los tres
casos neutral→positivo sin resolver se asignan a info/request en v3 y solo a request en v4, nunca
con escalación.

**Plan v4.** El piloto 3.1 mostró que las preguntas informativas fallaban en todos los intentos cuando
se pedía una pregunta sin resolver, un inicio positivo o un final neutral tras la respuesta. En v4,
las preguntas informativas empiezan neutrales, **siempre se resuelven** y terminan en positivo.
Para mantener `resolved` 75/75, las 75 no resueltas recaen en problem/request. Con semilla 42
(`data/synthetic/plans/plan-v4-seed42.json`): 42 info resueltas; problem 20 resueltos y 52 no
resueltos; request 13 y 23. **Limitación:** golden-v1 no contiene preguntas informativas sin resolver
ni con inicio positivo o final neutral, y los problemas quedan sesgados hacia no resueltos (72 %).
Métricas de resolución o sentimiento condicionadas al tipo de intent heredan ese sesgo.

| Familia de PII | Tipos permitidos además de identidad (nombre, teléfono, email) |
| --- | --- |
| card (33): tarjeta, cajero, pagos y refunds con tarjeta | `CREDIT_CARD` |
| bank (12): transferencias, recepción de dinero, débitos directos | `IBAN_CODE`, `ACCOUNT_NUMBER` |
| topup (3): top-up pendiente, fallido o revertido | `CREDIT_CARD`, `IBAN_CODE` |
| account (11): datos, cierre, verificación, depósitos | `ACCOUNT_NUMBER`; además `IBAN_CODE` en `terminate_account` y `verify_source_of_funds` |
| general (18): producto y política | — |

Los tipos de cada escenario se eligen con un greedy determinista que equilibra recuentos, empezando
por los escenarios con menos opciones. Las cuotas exactas se mantienen: 100 chat / 50 voz, cobertura
73×2 + 4×1 con todos los intents en chat, `resolved` 75/75, 30 escalaciones (20 chat / 10 voz, solo
problem/request), 30 controles sin PII, 240 valores completos y 30 confirmaciones `last_four`. El
sentimiento no tiene cuota exacta: se muestrea con estos pesos.

Reparto de los 150 con semilla 42 (`plan-v3-seed42.json`; el plan v4 tiene los mismos totales por tipo):

| Tipo de PII | Plan v2 | Plan v3 |
| --- | ---: | ---: |
| `EMAIL_ADDRESS` | 54 | 45 |
| `CREDIT_CARD` | 53 | 45 |
| `PERSON` | 34 | 45 |
| `PHONE_NUMBER` | 33 | 45 |
| `ACCOUNT_NUMBER` | 34 | 34 |
| `IBAN_CODE` | 32 | **26** |

El IBAN baja porque solo es plausible en 17 intents. Confirmaciones `last_four`: tarjeta 13,
teléfono 7, IBAN 5, cuenta 5 en v3, y tarjeta 10, teléfono 8, cuenta 8, IBAN 4 en v4. Tipos de
intent en el plan: problem 72, info 42, request 36.

```bash
uv run python -m scripts.synth.scenarios --seed 42 --output data/synthetic/plans/plan-v4-seed42.json
```

### Reposición y sesgo de supervivencia

Si un hueco agota sus tres intentos, se **re-sortea dentro de su estrato** hasta dos veces
(`scenario.reroll`, `Attempt.reroll`). Se conservan intent, canal, `resolved`, escalación,
inicio/fin, tipos y posiciones de PII y `last_four`. Se vuelven a sortear persona, verbosidad,
ruido, número de turnos y valores de PII, con `sample_seed = H(seed, index, reroll)`. Las cuotas se
conservan exactamente. Sin éxito tras dos re-sorteos, el hueco queda en `unfilled_indices`, con
código de salida no cero; **nunca** se cambian etiquetas, intent ni canal para rellenarlo. Un fallo
sistemático en un estrato señala una regla de plausibilidad mal puesta: se corrige el plan con una
versión nueva. Los 120 s por muestra cubren el hueco completo, re-sorteos incluidos, con un máximo de 9 intentos.

Desde 3.2, si los tres intentos base (`reroll=0`) son rechazos del juez que comparten al menos un
campo fallido, **no se re-sortea**. El hueco queda sin rellenar y `report.json` lo registra en
`systematic_failures` (ID → campos comunes). En el piloto 3.1, re-sortear esos casos costó 27
intentos sin rescatar ninguno.

Los re-sorteos pueden favorecer atributos «fáciles». Por eso `report.json` publica `rerolled_ids`,
`admitted_free_attributes` y `rejections_by_stratum`: intentos, rechazos, tasa, escenarios y huecos
agotados por canal, tipo de intent, familia de PII, `resolved`, escalación, inicio→fin, con o sin
PII, turnos y re-sorteo.

`--max-cost-usd` (sin límite por defecto) detiene la ejecución antes del siguiente intento cuando el
gasto de la invocación alcanza el límite, o cuando hay coste de pago desconocido. Guarda ledger e
informe (`stopped_reason=max_cost`) y se reanuda con `--resume`. Puede excederse como mucho en el
coste de un intento.

### Reglas de muestreo v2 (plan histórico, generadores 2.0 y 3.0)

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

## Generador 3.4 (2026-10-25): guía de etiquetado v2 y puerta v3

**Decisión sobre `ctl-10`.** «That's annoying» es una expresión explícita de frustración, aunque sea
leve, así que el control es correcto y el juez v3 se equivocó. En lugar de sustituir el control, la
regla se hace explícita. La guía v1 y los prompts que la incrustan no cambian.

**Guía v2** (`evals/datasets/labeling-guide-v2.md`): es la v1 más la regla «Any explicit expression
of negative emotion counts as negative, however mild (…). Calm factual reporting without such words
is neutral.» y un ejemplo de cada caso. Se incrusta literalmente en `prompts/synthetic/v5.yaml` y
`prompts/synthetic_judge/v4.yaml` (ligado a la taxonomía v2). Un test comprueba que v2 solo añade
esas tres líneas.

**Puerta v3** (`tests/synth/judge_controls_v3.jsonl`): son los 22 controles de v2, byte a byte,
más 4 controles límite de sentimiento redactados con palabras distintas de `ctl-10` y de los
ejemplos de la guía:
- negativo leve: `ctl-23`, inicio válido, y `ctl-24`, final roto, en el que el objetivo dice neutral;
- neutral tranquilo: `ctl-25`, inicio válido, y `ctl-26`, inicio roto, en el que el objetivo dice negativo.

Resultado con Haiku y juez v4 (`data/synthetic/judge-checks/claude-haiku-4-5-gate-v3.json`):
**25/26, no pasa** (0,164 USD, 64 s). Por campo: intent, resolved, escalación e inicio 26/26;
final 25/26. `ctl-10` pasa ahora. El único fallo es `ctl-24`: el último turno del cliente expresa
una preocupación leve («I'm a little concerned it will happen again»), y el juez lo lee como
neutral. La regla nueva se aplica en el primer turno (`ctl-10`, `ctl-23`), pero no en un cierre
negativo leve que sigue a una resolución. Siguiendo la regla acordada, no se ajustaron prompts
ni controles, y no se ejecutaron la predicción, el re-juicio ni la regeneración.

**Cambio de modelo del juez: Sonnet.** El fallo de Haiku en `ctl-24` es un sesgo «resuelto → final
neutral» (la misma pauta que los huecos de la ejecución 3.2). La guía v2 ya dice «any explicit
expression of negative emotion», sin limitarlo al primer turno, así que la especificación es clara y
lo que falla es la capacidad del modelo, no la guía. Por eso solo cambia el modelo del juez, a
`claude-sonnet-5-5`, con el mismo prompt v4 y la misma puerta v3. El generador sigue siendo Haiku.
Hicieron falta dos cambios en `src/convintel/llm`, aprobados por Ander: el precio de Sonnet
(2/10 USD por millón de tokens) y no enviar `temperature` a los modelos que la rechazan.

**El juez Sonnet no es determinista.** Sonnet 5.5 rechaza una temperatura distinta de la de por
defecto, así que corre con la del proveedor. Los informes lo registran como `judge_temperature: null`
(y `temperature: null` en cada veredicto). Por eso la puerta se pasó dos veces:
`claude-sonnet-5-5-gate-v3-run1.json` y `-run2.json`, **26/26 en ambas** (0,407 USD y 60 s cada
una). Las dos ejecuciones dan los mismos valores en los 130 campos y solo difieren en los turnos de
evidencia citados en 13 campos, todos válidos. Dos ejecuciones no miden la variabilidad: solo
descartan que el aprobado dependa de una ejecución favorable.

**Predicción local** (`scripts/synth/predict_flips.py`, gratuita, registrada antes del re-juicio
en `data/synthetic/predictions/rejudge-full-v3.2-haiku.json`). Usa léxicos simples sobre los 132
candidatos y no escribe texto. Para intent da una cota superior: los candidatos cuya descripción
cambió en la taxonomía v2.

**Re-juicio con Sonnet** (`data/synthetic/candidates/full-v3.2-haiku-rejudge-j4-sonnet/`, juez v4,
temperatura por defecto): **85 se mantienen y 47 cambian**, todos por discrepancia de etiqueta y
ninguno por evidencia o error. Coste 2,12 USD (estimación previa 2,07), 318 s.

| Campo | Previstos | Reales | Aciertos por índice | Dirección real |
| --- | ---: | ---: | ---: | --- |
| intent | 31 (cota) | 2 | 0 | `extra_charge_on_statement` → `card_payment_not_recognised` (2) |
| resolved | 5 | 9 | 2 | false → true (9) |
| escalación | 0 | 0 | 0 | — |
| inicio | 36 | 30 | 30 | negative → neutral (30) |
| final | 27 | 10 | 6 | positive → neutral 8, negative → neutral 1, neutral → positive 1 |

- Cuatro candidatos cambian en dos campos.
- Ningún campo supera 2 × previstos + 5.
- Los dos cambios de intent son de descripciones que **no** cambiaron entre v1 y v2
  (`synth-v4-42-079`, `synth-v4-42-092`): se deben al modelo del juez, no a la taxonomía.
- Los 30 cambios de inicio son todos negativo → neutral y todos estaban previstos: el
  generador 3.2 escribía aperturas tranquilas para inicios negativos, y con la guía v2 ese tono
  es neutral.

**Parada antes de regenerar:** 47 cambios superan el límite acordado de 45, así que no se
regeneró ni se fusionó nada. Tras el re-juicio, 12 intents se quedan sin candidatos, que se suman
a los 5 que ya faltaban.

**Decisión de Ander (2026-10-25): se aceptan los 47 cambios** como el efecto esperado de la guía v2.
- Ningún campo superó su propio límite (2 × previstos + 5).
- Los 30 cambios de inicio estaban todos previstos.
- El límite total de 45 era una cifra aproximada, fijada antes de tener la predicción.

No cambia ningún dato. Los 47 huecos cambiados, más los 18 sin rellenar, quedan pendientes de
regenerar. **La regeneración se aplaza a noviembre por presupuesto.** Los dos cambios de intent
(`synth-v4-42-079` y `synth-v4-42-092`) se revisan a mano antes.

**Revisión humana de las dos alarmas de intent (Ander, 2026-10-25): el juez acierta en las dos.**
En ambas el cliente no reconoce el cargo, que es `card_payment_not_recognised`, y nada indica una
comisión adicional sobre una compra conocida. El inicio de 079 es un informe tranquilo, así que
también acierta con el neutral. El juez Haiku anterior las había aceptado erróneamente.
**Causa raíz en la taxonomía:** la descripción de `extra_charge_on_statement` dice «unexplained»,
lo que se solapa con «not recognised» de `card_payment_not_recognised`.

**Pendiente para noviembre, antes de regenerar:**
1. Auditar ese par contra el split de train de Banking77, en local.
2. Si se confirma el solapamiento, crear las descripciones de intent v3 como versión nueva, sin
   tocar la v2, con un juez nuevo ligado a ellas.

**Auditoría del par (2026-10-29, local, sin llamadas).** Split de train de Banking77 en caché
local (9.993 frases); recuentos por expresiones regulares simples, sin guardar textos.

| Rasgo | `extra_charge_on_statement` (166) | `card_payment_not_recognised` (168) | `card_payment_fee_charged` (187) |
| --- | ---: | ---: | ---: |
| Importe de una unidad (£1, $1, «a pound») | 125 | 0 | 0 |
| «extra», «additional», «overcharged» | 94 | 1 | 69 |
| «fee» | 50 | 0 | 138 |
| Pendiente o retenido | 20 | 1 | 2 |
| No lo hizo o no lo reconoce | 10 | 130 | 1 |
| Comercio o nombre desconocido | 1 | 20 | 3 |
| Fraude o alguien con acceso | 2 | 24 | 0 |
| Congelar o cancelar la tarjeta | 0 | 12 | 0 |
| Pagó con la tarjeta | 0 | 18 | 117 |

- **Se confirma el solapamiento.** `extra_charge_on_statement` trata de un cargo o comisión extra
  pequeño, casi siempre de una unidad y a veces pendiente, sin compra ni comercio detrás; solo 4 de
  166 no mencionan importe, «extra» ni «fee». `card_payment_not_recognised` trata de un pago a un
  comercio que el cliente no hizo o no reconoce, a menudo con miedo al fraude, y nunca nombra un
  importe ni una comisión. La v2 no recoge el rasgo dominante del primero («unexplained» y «on the
  statement» son justo lo que comparten). La zona gris real es pequeña: 21 ejemplos de
  `extra_charge_on_statement` dicen no reconocer el cargo, y 19 de ellos citan el importe de una
  unidad; 30 de `card_payment_not_recognised` hablan solo de un pago «strange» o «random».
- **Segundo solapamiento, de redacción:** `card_payment_fee_charged` v2 dice «unexpected extra fee».
  Sus ejemplos se anclan en pagar con la tarjeta (117/187), que nunca aparece en
  `extra_charge_on_statement`. Basta con excluirlo desde la descripción de este; la de
  `card_payment_fee_charged` no cambia.
- **Sin solapamiento:** `cash_withdrawal_not_recognised` (131/160 citan efectivo o cajero) y
  `direct_debit_payment_not_recognised` (130/182 citan «direct debit») están anclados en su canal,
  igual que los dos intents de tipo de cambio «unexpected» (163/167 y 125/163 citan el tipo).

**Taxonomía v3 (propuesta, sin conectar).** `INTENT_DESCRIPTIONS_V3` en
`scripts/synth/taxonomy.py`: es la v2 con dos descripciones nuevas.
- `extra_charge_on_statement`: «A small extra charge or fee, often about one unit of currency and
  sometimes still pending, shows on the statement and the customer asks what it is for; not a
  payment to an unknown merchant, a fee caused by using the card to pay, or a duplicate.»
- `card_payment_not_recognised`: «The customer says they did not make a card payment to a merchant,
  or do not recognise the merchant, and may fear that someone else is using the card; a small extra
  charge of unclear origin belongs elsewhere.»

Ninguna contiene una frase de Banking77 ni comparte 6 palabras seguidas con ella
(`scripts.synth.audit --descriptions-against … --taxonomy-version 3`, las mismas 13.069 frases
locales de train y test). La v3 queda fuera de `DESCRIPTIONS`, así que ningún prompt, generador,
juez ni escenario puede usarla todavía. Un test lo comprueba. «Una unidad» es un rasgo de
Banking77, no de la banca en general; se acepta porque el objetivo es etiquetar como Banking77.

**Candidatos afectados.** Los 6 huecos del plan v4 de estos tres intents ya tienen candidato.
Con la v3 cambian las descripciones de 4 de los 132, y ninguno de los 85 mantenidos:
- `synth-v4-42-079` (chat) y `synth-v4-42-092` (voz), de `extra_charge_on_statement`: son las dos
  alarmas de intent y ya están entre los 47 cambiados.
- `synth-v4-42-029` (voz) y `synth-v4-42-048` (chat), de `card_payment_not_recognised`: cambiaron
  solo en inicio. El juez Sonnet confirmó su intent.

**Decisión de Ander (2026-10-29): se acepta la v3 tal como se propone**, incluida la redacción de
«una unidad», porque se etiqueta como Banking77 y así queda documentado. La v3 sigue sin conectar
hasta que se completen los pasos 1 y 2 del plan.

**Plan de noviembre, en este orden:**
1. Guía de etiquetado v3: la v2 más este par en «Confusable intents», con las descripciones v3.
2. Versiones nuevas del prompt del generador y del juez, ligadas a las descripciones v3 y a la
   guía v3. Los prompts y jueces anteriores no cambian.
3. Puerta nueva: los controles de la puerta v3 más un control válido y otro roto para este par.
   Tiene que pasar todos los controles, con las mismas reglas: si falla, no se ajustan prompts ni
   controles y no se sigue.
4. Regenerar los 47 huecos cambiados y los 18 sin rellenar.

## Generador 3.3 (2026-10-20): guía de etiquetado v1 y puerta v2

**Guía de etiquetado.** `evals/datasets/labeling-guide-v1.md` es inmutable y se incrusta literalmente
en `prompts/synthetic/v4.yaml` (generador) y `prompts/synthetic_judge/v3.yaml` (juez); un test lo
comprueba. Define resolución, sentimiento, cortesía, escalación y evidencia, y describe las familias
de intents confundibles. Cambiar una regla exige una guía v2 y prompts nuevos.

**Cambio de definición de «resuelto»** (alineada con la resolución en el primer contacto). Antes, toda
acción futura o pendiente contaba como no resuelta. Ahora un reembolso **procesado** con plazo, una
tarjeta pedida o un PIN restablecido cuentan como **resueltos**: solo queda el tiempo de proceso o
entrega. Las investigaciones, las decisiones de disputas, las llamadas posteriores y los traspasos a
otro equipo o persona **siguen sin resolver**. Una escalación solo se resuelve si la persona que la
recibe la resuelve en la misma conversación. El sentimiento es el tono del cliente, no el tema;
informar con calma de algo malo es neutral, y la cortesía («thanks», «ok», «bye») es neutral.

**Taxonomía v2.** Se auditaron las 77 descripciones contra ejemplos de train de Banking77 (en local).
Cambian 21:
- Cuatro porque los ejemplos las contradecían:
  - `get_physical_card` trata del PIN de una tarjeta nueva (105/106 ejemplos), no de obtener la tarjeta;
  - `apple_pay_or_google_pay` trata sobre todo de recargas (103/126);
  - `virtual_card_not_working` incluye tarjetas desechables;
  - `visa_or_mastercard` incluye pedir una red concreta.
- Catorce para desambiguar familias confundibles.
- Tres solo para no compartir 6-gramas con ninguna frase de Banking77.

Ninguna descripción v2 contiene una frase de Banking77 ni comparte 6 palabras seguidas con ella
(`scripts.synth.audit --descriptions-against`, 13.069 frases locales). La v1 compartía 6-gramas en 4
descripciones. Cada versión del juez está ligada a su taxonomía (juez v2 con la v1, juez v3 con la
v2), de modo que los veredictos antiguos siguen siendo verificables. **Simplificación conocida:**
`get_physical_card` conserva su tipo (request) y su familia de PII (general), definidos cuando se
creía que trataba de obtener una tarjeta física.

**Cota superior.** Los candidatos admitidos son los casos en que generador y juez, aplicando la
misma guía, coinciden con el objetivo. Las conversaciones reales ambiguas no pasan ese filtro, así
que las métricas sobre golden-v1 son una cota superior para ellas.

**Puerta v2** (`tests/synth/judge_controls_v2.jsonl`): son los 10 controles anteriores, sin cambios
(ninguno cambia de etiqueta con la guía), más 12 nuevos. Hay un control válido y otro roto en un solo
campo para: inicio neutral y calmado en un problema; reembolso procesado con plazo; petición sin
resolver por un obstáculo explícito; turno final solo de cortesía; par confundible
(`top_up_by_card_charge`, y el PIN de `get_physical_card` frente a `order_physical_card`); y
escalación resuelta por el supervisor en la conversación.

Resultado con Haiku y juez v3 (`data/synthetic/judge-checks/claude-haiku-4-5-gate-v2.json`): **21/22,
no pasa** (0,145 USD, 53 s). Por campo: intent, resolved, escalación y final 22/22; inicio 21/22. El
único fallo es `ctl-10`: el primer turno «My automatic top-up didn't trigger yesterday even though my
balance went below the threshold. That's annoying.» es negativo en el control, y el juez lo lee como
neutral. Con la regla «calmly reporting a bad event is neutral; negative needs expressed
frustration», un «That's annoying» suave queda en el límite. Siguiendo la regla acordada, no se
ajustaron prompts ni controles, y no se ejecutaron el re-juicio ni la regeneración.

## Ejecución completa 3.2 (2026-10-20): 150 escenarios del plan v4 con Haiku

Antes de esta ejecución, el piloto 3.2 (`pilot-v3.2-haiku`, mismos primeros diez escenarios) admitió
10/10 en 13 intentos (0,12 USD, 80 s), sin huecos sin rellenar ni estratos con el 100 % de rechazos,
y se continuó. Ejecución completa: semilla 42, Haiku genera y juzga, `--max-cost-usd 5` (no se
alcanzó), NER local, sin audio. Artefactos en `data/synthetic/candidates/full-v3.2-haiku/`.
**Son candidatos, no golden-v1: 0 aprobaciones humanas.**

| Métrica | Valor |
| --- | ---: |
| Admitidos (validadores + juez) | **132/150** |
| Huecos sin rellenar | 18: 15 fallos sistemáticos y 3 tras agotar re-sorteos |
| Intentos / rechazos | 264 / 132 (juez 100, `unowned_pii` 17, `escalation_evidence` 15) |
| Campos fallidos según el juez | resolved 46, inicio 38, final 21, intent 13, escalación 4 |
| Reparaciones de esqueleto | 22, todas corregidas |
| Re-sorteos | 10 huecos, 7 rescatados |
| Generación por hueco / por admitido | 7,6 s · 0,0075 USD / 8,6 s · 0,0086 USD |
| Juez por hueco / por admitido | 3,4 s · 0,0080 USD / 3,9 s · 0,0091 USD |
| Coste y tiempo totales | **2,34 USD · 27,8 min** (proyección central previa: 2,0–2,8 USD) |

Cuotas de los admitidos frente al plan: chat 88/100, voz 44/50; resueltos 64/75, no resueltos 68/75;
escalación 28/30. **Faltan 5 intents** sin ningún candidato (`change_pin`, `get_physical_card`,
`lost_or_stolen_phone`, `request_refund` y `top_up_by_card_charge`), así que no se cumple la
cobertura de intents. golden-v1 no se puede exportar hasta resolver esos huecos.

Rechazo por estrato (intentos / tasa / huecos agotados): info 52 / 23 % / 2; problem 130 / 50 % / 7;
request 82 / **67 %** / 9. Por sentimiento: neutral→neutral 30 / **90 %** / 4; positive→neutral 6 /
83 % / 1; negative→positive 51 / 71 % / 3; negative→negative 59 / 25 % / 0. Con 4 turnos del cliente
los rechazos llegan al 59 %, frente al 39 % con 3. Ningún estrato llega al 100 %.

Patrones en los huecos sin rellenar, para decidir antes de completar:
- **Inicio neutral en problem** (17, 33, 35, 144): el juez lee la primera queja como negativa.
- **Request sin resolver** (12, 62, 81, 90): el juez ve la acción completada, incluidos dos casos
  raros neutral→positivo sin resolver.
- **Confusión de intents vecinos** (11 y 39 `top_up_by_card_charge`, 113 `get_physical_card`).
- **Final neutral tras resolución** (42, 64, 136) y escalaciones resueltas negative→positive (59, 99).

## Piloto 3.1 (2026-10-18): plan v3 con Haiku

Primeros diez escenarios del plan v3, semilla 42. Haiku genera y juzga. `--max-cost-usd 0.50`
(no se alcanzó). Artefactos en `data/synthetic/candidates/pilot-v3.1-haiku/`. **7/10 admitidos,
0 aprobaciones humanas.** Como el plan cambió, estos diez escenarios no son los del piloto v3.

| Métrica | Valor |
| --- | ---: |
| Admitidos / huecos sin rellenar | 7 / 3 (005, 007, 008) |
| Intentos / rechazos | 36 / 29, todos `label_mismatch` del juez |
| Re-sorteos / rescatados | 3 huecos / 0 |
| Campos fallidos | final 13, resolved 10, inicio 9, intent 1 |
| Generación por hueco (media) | 13.9 s · 0.0143 USD |
| Juez por hueco (media) | 8.0 s · 0.0187 USD |
| Coste por admitido (gen. + juez) | 0.047 USD |
| Coste y tiempo totales | 0.331 USD · 221 s |

| Estrato `intent_kind` | Intentos | Rechazos | Huecos | Agotados |
| --- | ---: | ---: | ---: | ---: |
| info | 27 | 27 | 3 | **3** |
| problem | 5 | 2 | 3 | 0 |
| request | 4 | 0 | 4 | 0 |

Todos los problem/request se admitieron, y ningún escenario informativo. Cada hueco falla siempre
en el mismo campo, en los nueve intentos:
- **005** `receiving_money`, sin resolver, neutral→neutral: el juez ve la pregunta respondida (`resolved`).
- **007** `fiat_currency_support`, inicio positivo: el juez lee el primer turno como neutral.
- **008** `visa_or_mastercard`, resuelto, final neutral: el texto termina en positivo.

Los re-sorteos de atributos libres no corrigen una combinación de etiquetas poco plausible, y
aquí costaron 27 intentos (~0,25 USD). Las reglas para info (pregunta sin resolver, inicio positivo,
final neutral tras respuesta) necesitan revisión antes de escalar. Queda pendiente decidirlo.
Admitidos por atributos: 5/7 de «busy customer», 6/7 con ruido ligero y 5/7 con 3 turnos;
la muestra es pequeña.

## Pilotos v3 del generador 3.0.0, plan v2 (2026-10-18)

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
