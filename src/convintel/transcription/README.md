# ASR local con faster-whisper

`WhisperTranscriber` satisface el contrato congelado `Transcriber` y devuelve `Conversation`.
No es un operador de análisis ni devuelve `OperatorResult`. No hay pyannote, endpoints HTTP,
descargas de grabaciones ni declaraciones de métricas Prometheus en este módulo.

## Instalación y configuración

```bash
uv sync
```

El entorno normal permite importar ingesta/transcripción y ejecutar todas las unitarias sin
faster-whisper, CTranslate2 ni torch. El backend real se importa de forma lazy siguiendo el
patrón de T02. Para inferencia, tanto CPU como CUDA, instalar el grupo opt-in existente:

```bash
uv sync --group ml
```

Sin backend, una petición real falla con `TranscriptionError` y el comando de instalación.
Las opciones son argumentos de constructor, agrupados en el modelo Pydantic `WhisperSettings`:

```python
from pathlib import Path
from convintel.transcription import WhisperSettings, WhisperTranscriber

transcriber = WhisperTranscriber(WhisperSettings(model_path=Path("models/whisper-small")))
conversation = await transcriber.transcribe("/srv/audio/call.wav", "call-001")
```

Defaults: `small`, CPU `int8`, cuatro threads, idioma autodetectado. Opciones: `model_size`
(`tiny`, `base`, `small`, `medium`, `large-v3`), `model_path`, `cache_dir`, `device`,
`cpu_threads`, `language`, `stereo_speakers`. Los pesos deben corresponder a `model_size`.
No se añadieron ajustes globales ni variables de entorno. Follow-up T12:
**wire WhisperSettings to env** cuando se construya el endpoint/composición de la aplicación.

## Aprovisionamiento explícito: una vez, antes del servicio

Desde la raíz, este comando **sí descarga** el modelo `small` deliberadamente al directorio
conocido `models/whisper-small`, ignorado por Git. No se ejecuta desde runtime ni desde tests:

```bash
uv run --group ml python -c "from faster_whisper.utils import download_model; download_model('small', output_dir='models/whisper-small', use_auth_token=False)"
```

Pasar esa ruta mediante `WhisperSettings(model_path=Path("models/whisper-small"))`. En
producción, aprovisionar y montar el directorio de solo lectura antes de aceptar peticiones.
Si se omite `model_path`, solo se busca el modelo en la caché local, nunca en Internet.

Se exigen `model.bin`, `config.json`, `tokenizer.json` y `vocabulary.json` o `vocabulary.txt`
antes de construir el backend. `preprocessor_config.json` es opcional: el snapshot oficial
`small` no lo incluye y faster-whisper usa defaults locales. El tokenizer sí es obligatorio:
faster-whisper puede intentar descargarlo si falta incluso con `local_files_only=True`.
Búsqueda en caché y constructor siempre usan carga local; no se usan tokens Hugging Face
locales. Si faltan archivos obligatorios, se lanza un error de aprovisionamiento.

## Audio, roles y privacidad

Se aceptan rutas locales y URIs `file://` locales, WAV/MP3/FLAC/OGG mono o estéreo. PyAV
remuestrea a 16 kHz conservando los canales. La allowlist de demuxers/protocolos rechaza
playlists y solicitudes de red nativas; no se descargan URLs `RecordingUrl`. Entradas vacías,
inválidas o de más de dos canales fallan con `TranscriptionError`.

- Estéreo: izquierda → `customer`, derecha → `agent`, con `stereo_speakers` configurable
  según el sentido real de la llamada Twilio. Mono → `unknown`, sin inventar diarización.
- Un turno por segmento no vacío, orden cronológico estable, índices consecutivos y
  timestamps relativos al audio. Se permiten solapamientos.
- `asr_confidence = exp(avg_logprob)` acotada a [0,1]: aproximación geométrica de las
  probabilidades de tokens del segmento, no una probabilidad calibrada. No se fusionan segmentos.
- Silencio → conversación sin turnos. VAD usa el ONNX incluido en faster-whisper, sin descarga.
  El primer canal con texto determina el idioma.
- `redacted=False`. No registrar la conversación completa ni saltarse T02. `started_at`
  corresponde a la producción del resultado ASR aislado; T12 conservará el inicio y metadata
  originales, impondrá límites de audio y descargará las grabaciones autenticadas.

La carga se reutiliza por instancia. Decodificación e inferencia se ejecutan fuera del event
loop con `asyncio.to_thread`; un lock en el worker serializa el modelo incluso si se cancela
una coroutine. Cancelar la espera no interrumpe CTranslate2.

## CPU, CUDA y observabilidad

CPU `int8` es el default. `device="cuda"` solicita float16. Un `RuntimeError` o `ValueError`
en carga/iteración provoca un único reintento completo en CPU int8, descartando resultados
parciales y conservando CPU para futuras llamadas. El span registra `asr.fallback=cpu_int8`,
no la excepción original. Un fallo nativo que termine el proceso no es recuperable mediante
este mecanismo; la integración CUDA se ejecuta separadamente de CPU.

El span `asr.transcribe` incluye `conversation.id`, `asr.model`, `audio.duration_s`,
`asr.device`, `asr.realtime_factor`; nunca texto, rutas, URLs ni eventos de excepción.
Los fallos marcan estado ERROR. Se observa el histograma compartido
`convintel.telemetry.metrics.ASR_REALTIME_FACTOR`, etiqueta `model`, una vez por transcripción
exitosa y nunca en fallo. No se declara ni registra otro histograma.

RTF = (decodificación + carga + inferencia) / duración real del archivo. El trabajo de ambos
canales se divide por una sola duración, no por el doble. Cold incluye carga; warm reutiliza
la instancia. No incluye espera por el lock. Metadata contiene modelo, dispositivo efectivo,
duración y RTF. Un caso de prueba no demuestra el SLO p95 < 0.5.

## Pruebas y benchmark reproducible

Las unitarias generan tonos mono/estéreo y simulan el backend, con sockets bloqueados,
spans en memoria y comprobaciones del contador/suma Prometheus. Verifican también imports
lazy, pesos/tokenizer ausentes, errores lazy, fallback CUDA y aislamiento de `.env`.

La integración usa la fixture pública JFK versionada en `tests/fixtures/audio/jfk.wav`.
`PROVENANCE.md` documenta fuente, dominio público, conversión y hashes. `generate.py` crea
localmente diez segundos de estéreo: [0,5) del discurso a la izquierda, [6,11) a la derecha,
con silencio en la otra mitad de cada canal. Son fragmentos del mismo orador, no dos personas.
No se requiere TTS, red ni descargas durante ninguna prueba.

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv lock --check
uv sync --group ml
uv run --group ml pytest -m 'integration and not gpu' tests/transcription -s -rs
uv run --group ml pytest -m 'integration and gpu' tests/transcription -s -rs
```

Las integraciones comprueban turnos de ambos canales y muestran RTF cold/warm, dispositivo
solicitado/efectivo, cuatro threads y `language=en`. Sin `ml`, se omiten mediante
`pytest.importorskip`; sin pesos preaprovisionados en `models/whisper-small`, se omiten con
motivo explícito. `--asr-model-path=/ruta/local` permite otra ruta; si es incompleta, falla.

## Resultados reales (2026-10-04)

WSL2/Linux, Python 3.12.3, Ryzen 9 8940HX, RTX 5060 Laptop (sm_120, 8151 MiB), driver 592.82.
Backend faster-whisper 1.2.1 / CTranslate2 4.8.2, cuatro threads CTranslate2, idioma `en` y
fixture JFK estéreo de diez segundos. Una medición cold/warm por ejecución independiente:

| Solicitud | Dispositivo efectivo | Cold RTF | Warm RTF |
| --- | --- | --- | --- |
| CPU int8 | CPU | 0.339061 | 0.200608 |
| CUDA float16, entorno sin ajuste de loader | CPU (fallback) | 0.363580 | 0.192315 |
| CUDA float16, rutas cuBLAS/cuDNN explícitas | CUDA | 0.839915 | 0.036978 |

En el primer intento CUDA, CTranslate2 detectó una GPU, anunció float16 y cargó el modelo,
pero al iterar segmentos lanzó exactamente:
`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`.
El fallback funcionó; sus tiempos no son rendimiento GPU. Las bibliotecas ya estaban
instaladas por torch en `ml`, pero no estaban en la ruta del loader de CTranslate2.
No fue un fallo demostrado de kernels sm_120.

El segundo intento pasó en CUDA con este ajuste limitado al proceso, sin instalar paquetes
adicionales ni modificar configuración global (rutas para el entorno Python 3.12 del repo):

```bash
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
uv run --group ml pytest -m 'integration and gpu' tests/transcription --asr-model-path=models/whisper-small -s -rs
```

Ambos canales produjeron turnos en los tres casos. Cold es la primera inferencia de una
instancia, incluyendo carga del modelo y VAD; el módulo faster-whisper ya fue importado por
`pytest.importorskip`. Warm reutiliza la instancia. No es una medición p95 ni una garantía
universal de compatibilidad GPU; solo se verificó esta combinación de hardware y versiones.

## Dependencias justificadas

| Paquete | Grupo | Motivo |
| --- | --- | --- |
| faster-whisper | ml | Backend real CTranslate2, opcional y lazy; unitarias no lo necesitan. |
| av | base | Decodificación/remuestreo real de audio con allowlist, también en unitarias sin ml. |
| numpy | base | PCM planar y separación de canales, usados directamente sin ml. |
| twilio | base | Validador oficial de firmas webhook, usado sin servicios externos. |

CTranslate2 y ONNX Runtime son transitivas de faster-whisper. No se necesita soundfile.
Se conservan las dependencias OTel API/SDK de main sin downgrade, y no se añaden torch/CUDA
fuera del grupo `ml` existente. Las dependencias y el lock se gestionan con `uv add`/`uv lock`.
