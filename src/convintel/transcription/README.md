# ASR local con faster-whisper

`WhisperTranscriber` satisface el protocolo congelado `Transcriber` y devuelve `Conversation`.
No es un operador de análisis ni devuelve `OperatorResult`. No hay pyannote, endpoints HTTP,
descargas de grabaciones, exporters OTel ni declaraciones de métricas Prometheus en este módulo.

## Uso y configuración

```python
from convintel.transcription import WhisperSettings, WhisperTranscriber

transcriber = WhisperTranscriber(WhisperSettings(model_size="small"))
conversation = await transcriber.transcribe("/srv/audio/synthetic.wav", "synthetic-call-001")
```

`WhisperSettings` es un modelo Pydantic inmutable de opciones explícitas, NO otro lector de
environment. Defaults: `small`, CPU `int8`, cuatro threads, idioma autodetectado. También
admite `tiny`, `base`, `medium`, `large-v3`, `model_path`, `cache_dir`, `language`, `device`
y `stereo_speakers`. Los archivos de `model_path` deben corresponder a `model_size`.

**Integración pendiente del propietario:** mover/conectar las opciones ASR a
`convintel.config.Settings` y añadir sus variables a `.env.example`. Esos archivos no están
en el alcance permitido de T01. Hasta entonces pasar las opciones explícitamente; no se
leen variables ASR ocultas ni se duplica la configuración global.

Se aceptan rutas locales y URIs `file://` locales, WAV/MP3/FLAC/OGG mono o estéreo. PyAV
remuestrea a 16 kHz conservando los canales. Otros esquemas, playlists/demuxers no permitidos,
archivos vacíos, inválidos o de más de dos canales fallan con `TranscriptionError`.
La allowlist del decodificador impide que un archivo local fuerce solicitudes de red nativas.
T12 debe imponer límites de tamaño/duración, descargar las grabaciones autenticadas y
combinar los resultados con la metadata y `started_at` originales de la llamada.

- Estéreo: izquierda → `customer`, derecha → `agent` por defecto. No es una garantía sobre
  la dirección de una llamada Twilio; cambiar `stereo_speakers=(Speaker.AGENT, Speaker.CUSTOMER)`
  cuando el flujo de llamada lo requiera.
- Mono: `unknown`. No se inventa diarización de varios interlocutores en un solo canal.
- Un turno por segmento no vacío; orden cronológico estable y reindexación desde cero,
  preservando el speaker y los timestamps relativos al audio. Se permiten solapamientos.
- `asr_confidence = exp(avg_logprob)` acotada a [0, 1]: media geométrica de probabilidades
  de tokens del segmento, no una probabilidad calibrada ni una media aritmética de palabras.
  No se fusionan segmentos en este lane.
- Silencio: conversación sin turnos. VAD usa el recurso ONNX incluido con faster-whisper,
  no un modelo descargado durante la petición. El primer canal con texto aporta el idioma.
- `redacted=False`: el llamante no debe registrar texto ni enviar la conversación a operadores
  antes de T02. `started_at` es la hora UTC de producción del resultado si se usa ASR aislado.

La carga del modelo es lazy y se reutiliza por instancia. Decodificación e inferencia se
hacen fuera del event loop con `asyncio.to_thread`. Un lock en el worker serializa el acceso
al modelo incluso si una coroutine es cancelada: cancelar la espera no interrumpe CTranslate2.

## Pesos: aprovisionamiento explícito, nunca descarga silenciosa

**Estos comandos sí descargan pesos; ejecutarlos deliberadamente antes de iniciar el servicio.**
No se ejecutan desde pruebas ni desde `WhisperTranscriber` en modo online.

```bash
uv run python -c "from faster_whisper.utils import download_model; download_model('small')"
```

Para desplegar un directorio autocontenido:

```bash
uv run python -c "from faster_whisper.utils import download_model; download_model('small', output_dir='/srv/models/whisper-small')"
```

Pasar `WhisperSettings(model_path=Path('/srv/models/whisper-small'))` usando `pathlib.Path`.
Montar los pesos de solo lectura. Se requieren `model.bin`, `config.json`, `tokenizer.json`,
`preprocessor_config.json` y `vocabulary.json` o `vocabulary.txt`. Se comprueban antes de
construir el modelo: faster-whisper puede intentar obtener un tokenizer de Internet cuando
falta `tokenizer.json`, incluso si se pidió carga local. Una caché incompleta produce un
error con el comando de aprovisionamiento; no provoca esa descarga implícita.

La búsqueda en caché y el constructor reciben siempre `local_files_only=True`; la búsqueda
no usa tokens Hugging Face locales. No se hacen llamadas a proveedores ni se descargan URLs
`RecordingUrl` dentro del transcriptor.

## CPU, Blackwell y observabilidad

CPU `int8` es el camino por defecto, sin torch ni CUDA. `device="cuda"` solicita float16;
un error de ejecución/capacidad en carga o iteración fuerza un único reintento completo en
CPU `int8`, descarta resultados parciales y conserva CPU para futuras llamadas. El span
registra `asr.fallback=cpu_int8`; no se registra la excepción original. La compatibilidad real
con RTX 5060 / sm_120 no fue validada en T01; no asumir que el wheel de CTranslate2 la soporta.

El span `asr.transcribe` contiene `conversation.id`, `asr.model`, `audio.duration_s`,
`asr.device` y `asr.realtime_factor`. No contiene texto, rutas de audio/modelos, URLs ni
excepciones de terceros. Los fallos marcan estado ERROR sin eventos con PII.

RTF = (segundos de decodificación + carga + inferencia) / duración real del archivo. Para
estéreo se mide el trabajo de ambos canales contra una sola duración, no contra el doble. La primera
llamada incluye carga (cold); las siguientes reutilizan el modelo (warm). No incluye espera
por el lock. Metadata guarda `asr_model`, `asr_device`, `audio_duration_s`,
`asr_realtime_factor`. **T05: emit ASR_REALTIME_FACTOR via telemetry.metrics.** No se crea
un segundo histograma aquí.

## Pruebas y benchmark

Las unitarias generan WAV estéreo de diez segundos y mono remuestreado; simulan
`WhisperModel` y fallos lazy/GPU. Los sockets de red están bloqueados. Las trazas se capturan
en memoria. Los tonos prueban canales y timestamps, no calidad de reconocimiento.

La integración real requiere pesos `small` preinstalados y un ejecutable local `espeak-ng`
o `espeak`, que genera dos frases sintéticas sin PII ni red. No se instala automáticamente.
El generador `tests/fixtures/audio/generate.py` produce diez segundos de voz estéreo PCM16,
y no sobrescribe archivos existentes.

```bash
uv run pytest -q
uv run pytest -m integration tests/transcription -s -rs
uv run pytest -m integration tests/transcription --asr-model-path=/srv/models/whisper-small -s -rs
```

La integración comprueba al menos un turno por speaker y muestra RTF cold/warm CPU int8 de
`small` con cuatro threads. Sin caché se omite explícitamente; una ruta explícita incompleta
falla. No hay número de RTF real disponible en este entorno: faltan pesos preinstalados y
eSpeak. No sustituirlo por el tiempo de los mocks ni afirmar que se cumple el SLO de 0.5.

## Dependencias para integrar fuera de este lane

El propietario debe incorporarlas con `uv add` y actualizar `uv.lock` antes de habilitar CI
con un entorno limpio. T01 solo pudo verificarlas en un entorno temporal, sin tocar el
manifiesto ni su política de seguridad:

| Paquete verificado | Grupo | Justificación |
| --- | --- | --- |
| `faster-whisper==1.2.1` | base | ASR CTranslate2 en CPU, sin dependencia de torch. |
| `av==15.1.0` | base | Decodificar/remuestrear con canales separados y allowlist, sin ffmpeg externo. |
| `numpy==2.3.3` | base | Representar y concatenar PCM planar para inferencia. |
| `twilio==9.8.0` | base | Algoritmo oficial de validación de firmas webhook, sin llamadas de red. |
| `opentelemetry-api==1.37.0` | base/T05 | Emitir spans sin configurar exporters. |
| `opentelemetry-sdk==1.37.0` | dev/T05 | Verificar spans con exporter en memoria, sin red. |

No se añadieron torch/CUDA ni pyannote; cualquier dependencia exclusivamente GPU futura
corresponde a `ml`. Coordinar versiones OTel con T05. La decisión CPU/offline y su ADR
formal también requieren al propietario, porque `docs/adr/` no está permitido para T01.
