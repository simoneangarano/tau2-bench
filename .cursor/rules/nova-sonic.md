# Amazon Nova Sonic Integration Guidelines

When working on Amazon Nova Sonic integration, follow this priority order for references:

## 1. Primary Source: AWS Documentation

Always check the official AWS documentation first:

- **Getting started**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
- **Code examples**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-code-examples.html
- **System prompts**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-system-prompts.html
- **Core concepts**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-core-concepts.html
- **Event lifecycle**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-event-lifecycle.html
- **Event flow**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-event-flow.html
- **Input events**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
- **Output events**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
- **Barge-in**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-barge-in.html
- **Turn-taking**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-turn-taking.html
- **Cross-modal**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-cross-modal.html
- **Language support**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html
- **Chat history**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-chat-history.html
- **Tool configuration**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-tool-configuration.html
- **Async tools**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-async-tools.html
- **Integrations**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-integrations.html

## 2. Secondary Source: Web Search and LiveKit implementation

If AWS docs don't have the answer, use web search or LiveKit `aws_sdk_bedrock_runtime` implementation as a reference if:
- AWS docs are insufficient
- Need to understand SDK-specific patterns
- Debugging event format issues

Location: `.venv/lib/python3.12/site-packages/livekit/plugins/aws/experimental/realtime/`

## Key Technical Notes

### Audio Format
- **Input**: 16kHz PCM16 mono (what we send to Nova)
- **Output**: 24kHz PCM16 mono (what Nova sends back)

### SDK
- Use `aws_sdk_bedrock_runtime` (Smithy-based SDK), NOT boto3
- The `receive()` method blocks at C-level and doesn't respect asyncio timeouts
- Handle receive in background tasks within the same event loop

### Event Format
- Use `contentName` (not `contentId`)
- Set `interactive: true` on content that should trigger responses
- System prompts also need `interactive: true`

### VAD / Turn Detection
- Nova Sonic uses Voice Activity Detection (VAD) to detect when user stops speaking
- **CRITICAL**: After sending speech audio, send ~1 second of silence to trigger VAD end-of-turn detection
- Don't immediately end the audio content block after speech - let VAD detect the turn end naturally
- Keep audio stream open for continuous conversation (like LiveKit does)
