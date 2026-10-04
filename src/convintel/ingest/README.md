# Ingesta offline y webhooks Twilio

Estas funciones devuelven los contratos congelados `Conversation`/`Turn`; no son endpoints
HTTP ni realizan descargas. Requieren `twilio` (validación oficial local de firmas) y
`opentelemetry-api` (trazas). La incorporación de dependencias al manifiesto corresponde al
propietario del repositorio, fuera de los paths permitidos de T01.

## Chat JSON

`normalize_chat(json_string)` acepta este esquema; los campos desconocidos se rechazan:

```json
{
  "id": "synthetic-chat-001",
  "started_at": "2026-01-01T12:00:00Z",
  "language": "es",
  "messages": [
    {"speaker": "customer", "text": "Necesito ayuda con el pedido de prueba."},
    {"speaker": "agent", "text": "Reviso el pedido de prueba."}
  ],
  "metadata": {"source": "synthetic"}
}
```

`id`, `started_at` con zona horaria y al menos un mensaje son obligatorios. `language`
es opcional (default `en`), al igual que `metadata` (`dict[str, str]`, default vacío).
`speaker` admite `customer`, `agent`, `bot` y `unknown`; el texto debe ser no vacío.
También se acepta un `ChatPayload` validado. Se conservan el orden de mensajes y los
metadatos; los índices se generan desde cero. El resultado NO está redactado.

## Twilio

Construir un `TwilioWebhook(url=..., params=..., signature=...)` con la URL externa exacta,
todos los campos originales del POST form-urlencoded y el encabezado `X-Twilio-Signature`.
Pasarlo a `normalize_twilio_sms` o `normalize_twilio_recording`. Ambas funciones llaman
obligatoriamente a `validate_twilio_webhook`: no existe un flag para omitir la firma.
El helper usa `Settings.twilio_auth_token`; sin token o firma válida lanza `ConvIntelError`.
Si `twilio_account_sid` está configurado, también se exige esa cuenta.

El futuro endpoint T12 debe reconstruir la URL usando configuración de proxy de confianza,
no encabezados forwarded arbitrarios; debe rechazar claves de formulario repetidas antes
de convertirlas a `dict[str, str]`. No descartar campos desconocidos antes de validar:
Twilio puede añadir campos y todos participan en la firma. No se aceptan webhooks JSON
con `bodySHA256` en este helper. La validación no ofrece deduplicación/replay protection;
T12 debe usar `MessageSid` o `RecordingSid` como clave idempotente.

- SMS entrante: exige `MessageSid` y `Body`, genera un turno `customer`, `channel=sms`.
  No normaliza MMS sin texto ni WhatsApp. No retiene números de teléfono en metadata.
- Grabación completada: exige `CallSid`, `RecordingSid`, `RecordingUrl` HTTPS sin credenciales,
  `RecordingStatus=completed`, duración finita no negativa y `RecordingChannels` 1 o 2.
  Devuelve `channel=voice`, sin turnos, con `audio_uri` y metadata de grabación.
  No descarga `RecordingUrl`: el futuro adaptador debe descargarla con autenticación,
  destinos permitidos y límites de tamaño, y pasar una ruta local al transcriptor.
- `started_at` puede proporcionarse explícitamente. Si falta, es la hora UTC de recepción,
  no una afirmación sobre la hora original de la llamada. No se persisten `From`/`To`.

Los errores de validación no incluyen el payload original. Los spans `ingest.*` solo
registran IDs y estados, nunca texto, teléfonos, URLs ni eventos de excepción. El llamante
no debe registrar el objeto `Conversation` completo antes de la redacción PII de T02.
La asignación de roles estéreo es configurable en el transcriptor: depende del sentido
real de la llamada y no puede deducirse de `RecordingChannels`.
