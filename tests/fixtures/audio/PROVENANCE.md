# Procedencia de la fixture JFK

`jfk.wav` contiene los once segundos de discurso público usados por faster-whisper en
sus propias pruebas. No es una conversación de un cliente ni audio sintetizado.

- Orador/obra: John F. Kennedy, discurso inaugural del 20 de enero de 1961.
- Fuente exacta: https://raw.githubusercontent.com/SYSTRAN/faster-whisper/65882eee9f5cdbeeb2d877f1131d48cf241b327d/tests/data/jfk.flac
- Revisión upstream: `65882eee9f5cdbeeb2d877f1131d48cf241b327d` (tag `v1.2.1`).
- Prueba upstream: https://github.com/SYSTRAN/faster-whisper/blob/65882eee9f5cdbeeb2d877f1131d48cf241b327d/tests/test_transcribe.py
- Licencia del audio: **dominio público en Estados Unidos**, obra del Gobierno federal
  realizada en funciones oficiales, conforme a 17 USC §105. No se atribuye al audio la
  licencia MIT del código de faster-whisper.
- Referencia archivística: https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001
- Referencia de dominio público de la grabación:
  https://commons.wikimedia.org/wiki/File:JFK_inaugural_address.ogg

## Transformación e integridad

El FLAC original se decodificó con PyAV 16.1.0, remuestreando a mono de 16 kHz, formato
`s16`, y se guardó completo en WAV PCM16 little-endian mediante `wave`. Se conservaron
los once segundos, sin transcripción ni edición del discurso. El WAV ocupa 352044 bytes,
por debajo del límite de archivos del repositorio; el FLAC original ocupa 1152693 bytes.

SHA-256 del FLAC original:
`63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`

SHA-256 del WAV versionado:
`b9e1ae4e0837e7b99f05e4f61f70f5732320a56614ab4514d803fa85f9a563c4`

`generate.py` construye localmente un WAV estéreo de diez segundos:
- Izquierda: segundos [0, 5) de JFK seguidos de cinco segundos de silencio.
- Derecha: cinco segundos de silencio seguidos de los segundos [6, 11) de JFK.

Son dos fragmentos distintos del mismo orador histórico. Los roles customer/agent son
etiquetas de canal para probar la atribución estéreo, no una afirmación de diarización
real de dos personas. Ninguna prueba descarga audio, ejecuta TTS ni registra el discurso.
