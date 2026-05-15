using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;
using InceptorEngine;
using InceptorEngine.Input.Providers;

namespace InceptorEngine.Analytics // keep under InceptorEngine.* namespace for easy discovery
{
    // A lightweight service that owns OpenAI API calls and a small convo state.
    public class LLMConversationAnalyzer : MonoBehaviour
    {
        [Header("OpenAI Settings")] public string OpenAIKey;
        public string ModelChat = "gpt-5-mini";
        public string ModelChatReasoning = "gpt-5-mini";
        public string ModelTTS = "gpt-4o-mini-tts";
        public string ModelSTT = "gpt-4o-transcribe";
        public string Voice = "alloy";
        public string Language = "en";
        [TextArea] public string SystemPrompt = string.Empty;

        private const string OPENAI_TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech";
        private const string OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions";
        private const string OPENAI_STT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions";
        private const string OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses";

        [Serializable]
        public class ChatMessage { public string role; public string content; public ChatMessage(string r,string c){role=r;content=c;} }
        [Serializable]
        public class ChatCompletionRequest { public string model; public ChatMessage[] messages; public int max_tokens; public float temperature; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public object response_format; }
        [Serializable]
        public class ChatCompletionRequestV2 { public string model; public ChatMessage[] messages; public int max_completion_tokens; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public int? max_tokens; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public float? temperature; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public object response_format; }
        [Serializable]
        public class ReasoningConfig { public string effort; }
        [Serializable]
        public class ResponseTextConfig { [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public object format; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public string verbosity; }
        [Serializable]
        public class ResponsesRequest { public string model; public object input; public int max_output_tokens; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public ResponseTextConfig text; [JsonProperty(NullValueHandling = NullValueHandling.Ignore)] public ReasoningConfig reasoning; }
        [Serializable]
        public class Choice { public ChatMessage message; }
        [Serializable]
        public class ChatCompletionResponse { public Choice[] choices; }
        [Serializable]
        public class TTSRequest { public string model; public string input; public string voice; public string response_format; }
        [Serializable]
        public class TranscriptionResponse { public string text; }

        private readonly List<ChatMessage> _history = new();

        public void ResetConversation(string systemPrompt, string openingAssistantMessage = null)
        {
            _history.Clear();
            if (!string.IsNullOrEmpty(systemPrompt))
                _history.Add(new ChatMessage("system", systemPrompt));
            if (!string.IsNullOrEmpty(openingAssistantMessage))
                _history.Add(new ChatMessage("assistant", openingAssistantMessage));
        }

        // Returns the full conversation transcript as plain text
        public string GetTranscript()
        {
            var sb = new StringBuilder();
            foreach (var m in _history)
            {
                if (m.role == "system") continue; // skip system messages in transcript
                sb.Append(m.role == "user" ? "Therapist" : "Noa");
                sb.Append(": ");
                sb.AppendLine(m.content);
            }
            return sb.ToString();
        }

        public void AddUserUtterance(string text)
        {
            if (!string.IsNullOrWhiteSpace(text))
                _history.Add(new ChatMessage("user", text));
        }

        public IEnumerator GetChatResponse(Action<string> onDone, bool debug=false)
        {
            if (IsGpt5(ModelChat))
            {
                string direct = null;
                yield return RequestResponsesCompletion(ModelChat, _history.ToArray(), 2000, null, t => direct = t, debug);
                if (!string.IsNullOrEmpty(direct))
                    _history.Add(new ChatMessage("assistant", direct));
                onDone?.Invoke(direct ?? string.Empty);
                yield break;
            }

            var req = new ChatCompletionRequestV2
            {
                model = ModelChat,
                messages = _history.ToArray(),
                max_completion_tokens = 500,
                temperature = BuildTemperature(ModelChat, 0.7f)
            };

            var json = JsonConvert.SerializeObject(req);
            var body = Encoding.UTF8.GetBytes(json);
            using var request = new UnityWebRequest(OPENAI_CHAT_ENDPOINT, "POST");
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("Authorization", $"Bearer {OpenAIKey}");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                var raw = request.downloadHandler.text;
                var content = ExtractAssistantText(raw);
                if (string.IsNullOrEmpty(content))
                {
                    if (debug) Debug.LogWarning($"LLMConversationAnalyzer GetChatResponse: empty content. Raw response: {raw}");
                    string fallback = null;
                    yield return RequestResponsesCompletion(ModelChat, _history.ToArray(), 2000, null, t => fallback = t, debug);
                    content = fallback;
                }

                if (!string.IsNullOrEmpty(content))
                {
                    _history.Add(new ChatMessage("assistant", content));
                }
                onDone?.Invoke(content ?? string.Empty);
            }
            else
            {
                if (debug) Debug.LogError($"LLMConversationAnalyzer Chat failed: {request.error} -> {request.downloadHandler.text}");
                onDone?.Invoke(string.Empty);
            }
        }

        // Generic chat call with arbitrary messages (no history mutation)
        public IEnumerator RequestChatCompletion(ChatMessage[] messages, int maxTokens, float temperature, Action<string> onDone, bool debug=false, object responseFormat = null)
        {
            if (IsGpt5(ModelChatReasoning))
            {
                string direct = null;
                yield return RequestResponsesCompletion(ModelChatReasoning, messages, Math.Max(maxTokens, 2000), responseFormat, t => direct = t, debug);
                onDone?.Invoke(direct ?? string.Empty);
                yield break;
            }

            var req = new ChatCompletionRequestV2
            {
                model = ModelChatReasoning,
                messages = messages,
                max_completion_tokens = maxTokens,
                temperature = BuildTemperature(ModelChatReasoning, temperature),
                response_format = responseFormat
            };

            var json = JsonConvert.SerializeObject(req);
            var body = Encoding.UTF8.GetBytes(json);
            using var request = new UnityWebRequest(OPENAI_CHAT_ENDPOINT, "POST");
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("Authorization", $"Bearer {OpenAIKey}");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                var raw = request.downloadHandler.text;
                var content = ExtractAssistantText(raw);
                if (string.IsNullOrEmpty(content))
                {
                    if (debug) Debug.LogWarning($"LLMConversationAnalyzer RequestChatCompletion: empty content. Raw response: {raw}");
                    string fallback = null;
                    yield return RequestResponsesCompletion(ModelChatReasoning, messages, Math.Max(maxTokens, 2000), responseFormat, t => fallback = t, debug);
                    content = fallback;
                }
                onDone?.Invoke(content ?? string.Empty);
            }
            else
            {
                if (debug) Debug.LogError($"LLMConversationAnalyzer RequestChatCompletion failed: {request.error} -> {request.downloadHandler.text}");
                onDone?.Invoke(string.Empty);
            }
        }

        

        public IEnumerator TranscribeWav(byte[] wavBytes, Action<string> onDone, bool debug=false)
        {
            var form = new WWWForm();
            form.AddBinaryData("file", wavBytes, "audio.wav", "audio/wav");
            form.AddField("model", ModelSTT);
            form.AddField("language", Language);

            using var request = UnityWebRequest.Post(OPENAI_STT_ENDPOINT, form);
            request.SetRequestHeader("Authorization", $"Bearer {OpenAIKey}");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                var resp = JsonConvert.DeserializeObject<TranscriptionResponse>(request.downloadHandler.text);
                onDone?.Invoke(resp?.text?.Trim() ?? string.Empty);
            }
            else
            {
                if (debug) Debug.LogError($"LLMConversationAnalyzer STT failed: {request.error}");
                onDone?.Invoke(string.Empty);
            }
        }

        public IEnumerator Synthesize(string text, Action<byte[]> onDone, bool debug=false)
        {
            var req = new TTSRequest { model = ModelTTS, input = text, voice = Voice, response_format = "mp3" };
            var json = JsonConvert.SerializeObject(req);
            var body = Encoding.UTF8.GetBytes(json);

            using var request = new UnityWebRequest(OPENAI_TTS_ENDPOINT, "POST");
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("Authorization", $"Bearer {OpenAIKey}");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                onDone?.Invoke(request.downloadHandler.data);
            }
            else
            {
                if (debug) Debug.LogError($"LLMConversationAnalyzer TTS failed: {request.error}");
                onDone?.Invoke(null);
            }
        }

        // Utility for mic clip -> wav
        public static byte[] ClipToWav(AudioClip clip)
        {
            var samples = new float[clip.samples];
            clip.GetData(samples, 0);
            var intData = new short[samples.Length];
            for (int i=0;i<samples.Length;i++) intData[i] = (short)(Mathf.Clamp(samples[i],-1f,1f) * 32767);
            using var stream = new System.IO.MemoryStream();
            using var writer = new System.IO.BinaryWriter(stream);
            writer.Write("RIFF".ToCharArray());
            writer.Write(36 + intData.Length * 2);
            writer.Write("WAVE".ToCharArray());
            writer.Write("fmt ".ToCharArray());
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)1);
            writer.Write(clip.frequency);
            writer.Write(clip.frequency * 2);
            writer.Write((short)2);
            writer.Write((short)16);
            writer.Write("data".ToCharArray());
            writer.Write(intData.Length * 2);
            foreach (var s in intData) writer.Write(s);
            return stream.ToArray();
        }

        private static string ExtractAssistantText(string rawJson)
        {
            if (string.IsNullOrWhiteSpace(rawJson)) return string.Empty;
            try
            {
                var root = JObject.Parse(rawJson);

                // Standard chat.completions: choices[0].message.content (string)
                var contentToken = root.SelectToken("choices[0].message.content");
                if (contentToken != null)
                {
                    if (contentToken.Type == JTokenType.String)
                        return contentToken.Value<string>() ?? string.Empty;

                    // Some models return content as an array of parts
                    if (contentToken.Type == JTokenType.Array)
                    {
                        var sb = new StringBuilder();
                        foreach (var part in (JArray)contentToken)
                        {
                            if (part.Type == JTokenType.String)
                            {
                                sb.Append(part.Value<string>());
                                continue;
                            }
                            var text = part["text"]?.Value<string>()
                                       ?? part["content"]?.Value<string>();
                            if (!string.IsNullOrEmpty(text)) sb.Append(text);
                        }
                        return sb.ToString().Trim();
                    }

                    // Content returned as object
                    if (contentToken.Type == JTokenType.Object)
                    {
                        var text = contentToken["text"]?.Value<string>()
                                   ?? contentToken["content"]?.Value<string>();
                        return text ?? string.Empty;
                    }
                }

                // Some responses put text at choices[0].text
                var textToken = root.SelectToken("choices[0].text");
                if (textToken != null && textToken.Type == JTokenType.String)
                    return textToken.Value<string>() ?? string.Empty;

                // Streaming-style delta fallback if present
                var deltaToken = root.SelectToken("choices[0].delta.content");
                if (deltaToken != null && deltaToken.Type == JTokenType.String)
                    return deltaToken.Value<string>() ?? string.Empty;

                // Responses API fallback: output_text or find message in output array
                var outputText = root.SelectToken("output_text")?.Value<string>();
                if (!string.IsNullOrEmpty(outputText)) return outputText;

                // Iterate through output array to find the "message" type item
                var outputArray = root["output"] as JArray;
                if (outputArray != null)
                {
                    foreach (var item in outputArray)
                    {
                        // Look for type = "message" which contains the actual response
                        if (item["type"]?.Value<string>() == "message")
                        {
                            var contentArray = item["content"] as JArray;
                            if (contentArray != null)
                            {
                                var sb = new StringBuilder();
                                foreach (var part in contentArray)
                                {
                                    var text = part["text"]?.Value<string>()
                                               ?? part["content"]?.Value<string>();
                                    if (!string.IsNullOrEmpty(text)) sb.Append(text);
                                }
                                var result = sb.ToString().Trim();
                                if (!string.IsNullOrEmpty(result)) return result;
                            }
                        }
                    }
                }
            }
            catch (Exception)
            {
                // ignore parse errors and fall through
            }

            return string.Empty;
        }

        private IEnumerator RequestResponsesCompletion(string model, ChatMessage[] messages, int maxTokens, object responseFormat, Action<string> onDone, bool debug=false)
        {
            var inputList = new List<object>();
            for (int i = 0; i < messages.Length; i++)
            {
                var m = messages[i];
                if (string.IsNullOrWhiteSpace(m?.content)) continue;
                // Responses API uses "output_text" for assistant, "input_text" for user/system
                var contentType = m.role == "assistant" ? "output_text" : "input_text";
                inputList.Add(new
                {
                    role = m.role,
                    content = new[] { new { type = contentType, text = m.content } }
                });
            }

            var req = new ResponsesRequest
            {
                model = model,
                input = inputList,
                max_output_tokens = maxTokens,
                text = responseFormat == null ? null : new ResponseTextConfig { format = BuildTextFormat(responseFormat) },
                reasoning = new ReasoningConfig { effort = "low" }
            };

            var json = JsonConvert.SerializeObject(req);
            var body = Encoding.UTF8.GetBytes(json);
            using var request = new UnityWebRequest(OPENAI_RESPONSES_ENDPOINT, "POST");
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("Authorization", $"Bearer {OpenAIKey}");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                var raw = request.downloadHandler.text;
                Debug.Log($"[DEBUG] Responses API raw response: {raw}"); // TODO: Remove after debugging
                var content = ExtractAssistantText(raw);
                if (string.IsNullOrEmpty(content))
                    Debug.LogWarning($"LLMConversationAnalyzer Responses API: empty content. Raw response: {raw}");
                onDone?.Invoke(content ?? string.Empty);
            }
            else
            {
                if (debug) Debug.LogError($"LLMConversationAnalyzer Responses API failed: {request.error} -> {request.downloadHandler.text}");
                onDone?.Invoke(string.Empty);
            }
        }

        private static float? BuildTemperature(string model, float requested)
        {
            if (string.IsNullOrWhiteSpace(model)) return requested;
            // gpt-5* chat.completions only supports default temperature (1) and rejects explicit values
            if (model.StartsWith("gpt-5", StringComparison.OrdinalIgnoreCase))
                return null;
            return requested;
        }

        private static object BuildTextFormat(object responseFormat)
        {
            if (responseFormat == null) return null;
            try
            {
                var token = JToken.FromObject(responseFormat);
                var type = token["type"]?.Value<string>();
                var jsonSchema = token["json_schema"] as JObject;
                if (string.Equals(type, "json_schema", StringComparison.OrdinalIgnoreCase) && jsonSchema != null)
                {
                    var name = jsonSchema["name"]?.Value<string>();
                    var schema = jsonSchema["schema"] as JObject;
                    var strict = jsonSchema["strict"]?.Value<bool?>();
                    if (!string.IsNullOrWhiteSpace(name) && schema != null)
                    {
                        var mapped = new JObject
                        {
                            ["type"] = "json_schema",
                            ["name"] = name,
                            ["schema"] = schema
                        };
                        if (strict.HasValue) mapped["strict"] = strict.Value;
                        return mapped;
                    }
                }
            }
            catch (Exception)
            {
                // fall through to return original responseFormat
            }

            return responseFormat;
        }

        private static bool IsGpt5(string model)
        {
            return !string.IsNullOrWhiteSpace(model)
                   && model.StartsWith("gpt-5", StringComparison.OrdinalIgnoreCase);
        }
    }
}
