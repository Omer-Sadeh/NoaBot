using System;
using System.Collections;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;
using UnityEngine.UI;
using InceptorEngine.Clips.Interfaces;
using TMPro;

namespace InceptorEngine.Clips
{
    [CreateAssetMenu(menuName = "Inceptor/Clip/LLM Talking Clip", fileName = "LLMTalkingClip", order = 1)]
    [Serializable]
    public class TalkingClip : Clip
    {
    [Header("Clip Routing")]
    [SerializeField] private int _nextClip = 1;
    [Tooltip("Clip index to move to when all guidelines are completed.")]
    [SerializeField] private int _successClip = 2;

    [Header("Conversation Settings")]
    [SerializeField] private bool _showDebugLogs = true;
    [SerializeField] private string _aiVoice = "alloy";
    [SerializeField] private string _conversationLanguage = "en";
    [TextArea] [SerializeField] private string _systemPrompt = "";
    [TextArea] [SerializeField] private string _initialMessage = "";
    [TextArea] [SerializeField] private string _judgeSystemPrompt = "";

    [Serializable]
    public class GuidelineSection { public System.Collections.Generic.List<string> Items = new System.Collections.Generic.List<string>(); }
    [Header("Guidelines Judging")]
    [Tooltip("Sections of guidelines. Each section is a list of guideline strings.")]
    [SerializeField] private System.Collections.Generic.List<GuidelineSection> _guidelineSections = new System.Collections.Generic.List<GuidelineSection>();
    [Tooltip("Keeps track of remaining guidelines per section. Will be initialized from sections on build.")]
    [SerializeField] private System.Collections.Generic.List<GuidelineSection> _remainingGuidelines = new System.Collections.Generic.List<GuidelineSection>();
    [Tooltip("Zero-based index of current active guideline section.")]
    [SerializeField] private int _activeSectionIndex = 0;
    [TextArea] [SerializeField] private string _guidelinesJudgeSystemPrompt = "";
    
    [Header("Text Verdict tag")]
    [SerializeField] private string _verdictTextTag = "VerdictText";
    private TMP_Text _verdictTMP;
    [SerializeField] private Sprite _textWindowDefault;
    [SerializeField] private Sprite _textWindowSuccess;
    [SerializeField] private Sprite _textWindowWarning;

    private AudioSource _audioSource;
    private static bool _didInitialGreeting = false; // session-scoped guard
    private static bool _didInitializeGuidelines = false; // session-scoped guard
    private bool _hasUserClickedContinue = false;

        public override void Build(JObject clipData)
        {
            Debug.Log("TalkingClip Build: initializing from data");
            Initialize(clipData);
            // Try to read next clip from metadata
            try
            {
                var meta = clipData?["metadata"]; // may be null in editor-created assets
                if (meta != null && meta["nextClip"] != null)
                {
                    _nextClip = meta["nextClip"].Value<int>();
                }
            }
            catch (Exception e)
            {
                if (_showDebugLogs) Debug.LogWarning($"TalkingClip Build: failed to parse next clip metadata: {e.Message}");
            }

            // Always rebuild remaining guidelines from the current sections
            ResetRemainingGuidelinesFromSections();
        }

        protected override IEnumerator Run(UnityAction<int> onClipEnd)
        {
            InitializeGuidelinesForSession();
            var inceptor = Inceptor.GetAvailableInceptors();
            if (inceptor == null)
            {
                Debug.LogError("TalkingClip: Inceptor not found");
                onClipEnd?.Invoke(0);
                yield break;
            }

            CacheVerdictTMP();
            SetVerdictVisible(false);
            SetContinueButtonActive(false);

            var analyzer = FindOrCreateAnalyzer(inceptor);
            analyzer.Language = _conversationLanguage;
            analyzer.Voice = _aiVoice;

            // Determine if this is the first Talking run in the session
            bool isFirstTalking = !_didInitialGreeting;
            // Only reset the conversation on the first Talking run when a prompt or initial message is provided
            if (isFirstTalking && (!string.IsNullOrEmpty(_systemPrompt) || !string.IsNullOrEmpty(_initialMessage)))
            {
                analyzer.ResetConversation(_systemPrompt, _initialMessage);
            }

            string aiText = null;
            if (isFirstTalking && !string.IsNullOrEmpty(_initialMessage))
            {
                // First run: speak the configured initial message, do NOT fetch a new LLM response
                aiText = _initialMessage;
                _didInitialGreeting = true;
            }
            else
            {
                // Subsequent runs: continue the conversation based on the user's last utterance
                string userText = ListeningClip.LastUserText; // could be null for the opening
                if (!string.IsNullOrWhiteSpace(userText)) analyzer.AddUserUtterance(userText);

                PrintVerdict("Noa is thinking...");
                ApplyDefaultVisual();
                SetVerdictVisible(true);

                // Ask for AI response
                yield return inceptor.StartCoroutine(analyzer.GetChatResponse(t => aiText = t, _showDebugLogs));
            }

            SetVerdictVisible(false);
            if (_showDebugLogs) Debug.Log($"TalkingClip AI: {aiText}");

            bool conversationDone = true;
            bool guidelinesDone = true;
            string conversationVerdict = null;
            string guidelinesVerdict = null;
            bool guidelinesSuccess = false;
            bool allGuidelinesCompleted = false;

            if (!(isFirstTalking && !string.IsNullOrEmpty(_initialMessage)))
            {
                conversationDone = false;
                guidelinesDone = false;
                inceptor.StartCoroutine(JudgeConversationAsync(
                    analyzer,
                    v => conversationVerdict = v,
                    () => conversationDone = true
                ));
                inceptor.StartCoroutine(JudgeGuidelinesAsync(
                    analyzer,
                    v =>
                    {
                        guidelinesVerdict = v;
                        guidelinesSuccess = !string.IsNullOrEmpty(v);
                    },
                    allCompleted => allGuidelinesCompleted = allCompleted,
                    () => guidelinesDone = true
                ));
            }

            // TTS and play
            byte[] audioBytes = null;
            yield return inceptor.StartCoroutine(analyzer.Synthesize(aiText, b => audioBytes = b, _showDebugLogs));
            if (audioBytes != null)
            {
                yield return inceptor.StartCoroutine(PlayMp3(audioBytes));
            }
            if (_endDelay != 0) yield return new WaitForSeconds(_endDelay);

            // Wait for both conversation judging and guidelines judging to complete before proceeding
            while (!conversationDone || !guidelinesDone) yield return null;
            string verdictToPrint = !string.IsNullOrEmpty(guidelinesVerdict) ? guidelinesVerdict : conversationVerdict;
            if (!string.IsNullOrEmpty(verdictToPrint))
            {
                if (!string.IsNullOrEmpty(guidelinesVerdict))
                    Debug.Log($"TalkingClip Guidelines Verdict: {verdictToPrint}");
                else
                    Debug.Log($"TalkingClip Judge: {verdictToPrint}");
                PrintVerdict(verdictToPrint);
                ApplyVerdictVisual(!string.IsNullOrEmpty(guidelinesVerdict));
                SetVerdictVisible(true);
            }

            SetContinueButtonActive(true);

            // After the AI finishes speaking (PlayMp3)
            _hasUserClickedContinue = false;
            // Wait until the button is pressed
            while (!_hasUserClickedContinue) { yield return null; }

            SetVerdictVisible(false);
            SetContinueButtonActive(false);

            int next = _nextClip;
            if (guidelinesSuccess && allGuidelinesCompleted)
            {
                if (_showDebugLogs) Debug.Log("TalkingClip: all guidelines completed; moving to success clip.");
                next = _successClip;
            }
            else
            {
                if (_showDebugLogs) Debug.Log("TalkingClip: guidelines not fully completed; moving to next clip.");
                PrintVerdict("Your turn to speak.");
                ApplyDefaultVisual();
                SetVerdictVisible(true);
            }
            onClipEnd?.Invoke(next);
        }

        public void ProceedToNextClip() { _hasUserClickedContinue = true; }

        private IEnumerator PlayMp3(byte[] mp3)
        {
            // temp file for MP3 playback
            string path = System.IO.Path.Combine(Application.temporaryCachePath, "llm_talking.mp3");
            System.IO.File.WriteAllBytes(path, mp3);
            string url = "file://" + path;
            using (var www = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.MPEG))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                {
                    if (_audioSource == null)
                    {
                        // Try to find an existing object called "Noa" first
                        var NoaGo = GameObject.Find("Noa");
                        if (NoaGo != null)
                        {
                            _audioSource = NoaGo.GetComponent<AudioSource>() ?? NoaGo.AddComponent<AudioSource>();
                        }
                        else
                        {
                            // Fallback: create a dedicated GameObject to play audio
                            var go = new GameObject("TalkingClip_Audio");
                            _audioSource = go.AddComponent<AudioSource>();
                        }
                    }
                    var clip = DownloadHandlerAudioClip.GetContent(www);
                    _audioSource.clip = clip; _audioSource.Play();
                    yield return new WaitForSeconds(clip.length);
                }
                else Debug.LogError($"TalkingClip: audio load failed {www.error}");
            }
            if (System.IO.File.Exists(path)) System.IO.File.Delete(path);
        }

        private Analytics.LLMConversationAnalyzer FindOrCreateAnalyzer(Inceptor inceptor)
        {
            var ex = GameObject.FindFirstObjectByType<Analytics.LLMConversationAnalyzer>();
            if (ex != null) return ex;
            var go = new GameObject("LLMConversationAnalyzer");
            var svc = go.AddComponent<Analytics.LLMConversationAnalyzer>();
            svc.OpenAIKey = inceptor.Settings.OpenAIKey;
            return svc;
        }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void ResetGreetingFlag()
    {
        _didInitialGreeting = false;
        _didInitializeGuidelines = false;
    }

    private IEnumerator JudgeConversationAsync(Analytics.LLMConversationAnalyzer analyzer, System.Action<string> onVerdict, System.Action onDone)
    {
        var msgs = new System.Collections.Generic.List<Analytics.LLMConversationAnalyzer.ChatMessage>();
        if (!string.IsNullOrWhiteSpace(_judgeSystemPrompt))
        {
            // Allow placeholder replacement with current active guidelines section
            var sys = FillGuidelinesPlaceholder(_judgeSystemPrompt);
            msgs.Add(new Analytics.LLMConversationAnalyzer.ChatMessage("system", sys));
        }

        // use full transcript as a single user message
        var transcript = analyzer.GetTranscript();
        msgs.Add(new Analytics.LLMConversationAnalyzer.ChatMessage("user", transcript));

        string verdict = null;
        yield return analyzer.RequestChatCompletion(
            msgs.ToArray(),
            maxTokens: 200,
            temperature: 0.0f,
            onDone: v => verdict = v,
            debug: _showDebugLogs
        );

        onVerdict?.Invoke(verdict);
        onDone?.Invoke();
    }
    
    private void PrintVerdict(string verdict)
    {
        CacheVerdictTMP();

        if (_verdictTMP != null)
            _verdictTMP.text = verdict;
        else if (_showDebugLogs)
            Debug.LogWarning($"TalkingClip: No TMP_Text found with tag {_verdictTextTag}");
    }

    private void SetContinueButtonActive(bool active)
    {
        Debug.Log($"TalkingClip: Setting continue button active: {active}");

        var btn = FindContinueButton();
        if (btn != null)
        {
            var parent = btn.transform != null ? btn.transform.parent : null;
            var container = parent != null ? parent.gameObject : btn;
            container.SetActive(active);
        }
        else if (_showDebugLogs)
        {
            Debug.LogWarning("TalkingClip: No GameObject found with tag 'ContinueButton'");
        }
    }

    private GameObject FindContinueButton()
    {
        var btn = GameObject.FindWithTag("ContinueButton");
        if (btn != null) return btn;

        var allTransforms = Resources.FindObjectsOfTypeAll<Transform>();
        foreach (var t in allTransforms)
        {
            if (t != null && t.CompareTag("ContinueButton")) return t.gameObject;
        }

        return null;
    }

    private void SetVerdictVisible(bool visible)
    {
        CacheVerdictTMP();
        var target = _verdictTMP != null ? _verdictTMP.gameObject : GameObject.FindWithTag(_verdictTextTag);
        if (target != null)
        {
            var parent = target.transform != null ? target.transform.parent : null;
            var container = parent != null ? parent.gameObject : target;
            container.SetActive(visible);
        }
        else if (_showDebugLogs)
            Debug.LogWarning($"TalkingClip: No GameObject found with tag {_verdictTextTag}");
    }

    private void ApplyVerdictVisual(bool isGuidelinesVerdict)
    {
        CacheVerdictTMP();
        var target = _verdictTMP != null ? _verdictTMP.gameObject : GameObject.FindWithTag(_verdictTextTag);
        if (target == null) return;

        var parent = target.transform != null ? target.transform.parent : null;
        if (parent == null) return;

        var background = parent.Find("Background");
        if (background == null) return;

        var image = background.GetComponent<Image>();
        if (image == null) return;

        var sprite = isGuidelinesVerdict ? _textWindowSuccess : _textWindowWarning;
        if (sprite != null) image.sprite = sprite;
    }

    private void ApplyDefaultVisual()
    {
        CacheVerdictTMP();
        var target = _verdictTMP != null ? _verdictTMP.gameObject : GameObject.FindWithTag(_verdictTextTag);
        if (target == null) return;

        var parent = target.transform != null ? target.transform.parent : null;
        if (parent == null) return;

        var background = parent.Find("Background");
        if (background == null) return;

        var image = background.GetComponent<Image>();
        if (image == null) return;

        if (_textWindowDefault != null) image.sprite = _textWindowDefault;
    }

    private void CacheVerdictTMP()
    {
        if (_verdictTMP != null) return;

        var go = GameObject.FindWithTag(_verdictTextTag);
        if (go != null)
        {
            _verdictTMP = go.GetComponent<TMP_Text>();
            return;
        }

        var allTexts = Resources.FindObjectsOfTypeAll<TMP_Text>();
        foreach (var text in allTexts)
        {
            if (text != null && text.gameObject.CompareTag(_verdictTextTag))
            {
                _verdictTMP = text;
                break;
            }
        }
    }

    private void EnsureRemainingGuidelinesInitialized()
    {
        if (_remainingGuidelines == null || _remainingGuidelines.Count == 0)
        {
            ResetRemainingGuidelinesFromSections();
        }
    }

    private void ResetRemainingGuidelinesFromSections()
    {
        _remainingGuidelines = new System.Collections.Generic.List<GuidelineSection>();
        foreach (var section in _guidelineSections)
        {
            _remainingGuidelines.Add(new GuidelineSection
            {
                Items = new System.Collections.Generic.List<string>(section?.Items ?? new System.Collections.Generic.List<string>())
            });
        }
        _activeSectionIndex = 0;
    }

    private void OnValidate()
    {
        if (Application.isPlaying) return;
        ResetRemainingGuidelinesFromSections();
    }

    private void OnEnable()
    {
        if (Application.isPlaying)
        {
            ResetRemainingGuidelinesFromSections();
        }
    }

    private void InitializeGuidelinesForSession()
    {
        if (_didInitializeGuidelines) return;
        ResetRemainingGuidelinesFromSections();
        _didInitializeGuidelines = true;
    }

    private string BuildActiveSectionGuidelinesText()
    {
        EnsureRemainingGuidelinesInitialized();
        if (_remainingGuidelines == null || _remainingGuidelines.Count == 0) return string.Empty;
        if (_activeSectionIndex < 0 || _activeSectionIndex >= _remainingGuidelines.Count) return string.Empty;
        var section = _remainingGuidelines[_activeSectionIndex];
        if (section == null || section.Items == null || section.Items.Count == 0) return string.Empty;
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < section.Items.Count; i++) sb.AppendLine($"{i}: {section.Items[i]}");
        return sb.ToString().TrimEnd();
    }

    private string FillGuidelinesPlaceholder(string template)
    {
        if (string.IsNullOrEmpty(template)) return template;
        var guidelinesText = BuildActiveSectionGuidelinesText();
        return template.Replace("{guidelines_section}", guidelinesText);
    }

    private System.Object BuildGuidelinesResponseFormat()
    {
        return new
        {
            type = "json_schema",
            json_schema = new
            {
                name = "guidelines_evaluation",
                schema = new
                {
                    type = "object",
                    properties = new
                    {
                        completed_indexes = new
                        {
                            type = "array",
                            items = new { type = "integer" },
                            description = "Indexes of the guidelines that were completed by the therapist and Noa."
                        }
                    },
                    required = new[] { "completed_indexes" },
                    additionalProperties = false
                },
                strict = true
            }
        };
    }

    private IEnumerator JudgeGuidelinesAsync(Analytics.LLMConversationAnalyzer analyzer, System.Action<string> onVerdict, System.Action<bool> onAllCompleted, System.Action onDone)
    {
        EnsureRemainingGuidelinesInitialized();
        if (_remainingGuidelines == null || _remainingGuidelines.Count == 0)
        {
            onAllCompleted?.Invoke(true);
            onDone?.Invoke();
            yield break;
        }
        if (_activeSectionIndex < 0 || _activeSectionIndex >= _remainingGuidelines.Count)
        {
            onAllCompleted?.Invoke(AreAllGuidelinesCleared());
            onDone?.Invoke();
            yield break;
        }

        var activeSection = _remainingGuidelines[_activeSectionIndex];
        if (activeSection == null || activeSection.Items == null || activeSection.Items.Count == 0)
        {
            // advance to next section if exists
            if (_activeSectionIndex + 1 < _remainingGuidelines.Count)
                _activeSectionIndex++;
            onAllCompleted?.Invoke(AreAllGuidelinesCleared());
            onDone?.Invoke();
            yield break;
        }

        var transcript = analyzer.GetTranscript();

        string sysPrompt;
        if (!string.IsNullOrWhiteSpace(_guidelinesJudgeSystemPrompt))
        {
            sysPrompt = FillGuidelinesPlaceholder(_guidelinesJudgeSystemPrompt);
        }
        else
        {
            var defaultTemplate = "You are an evaluator. Determine which of the following guidelines have been clearly completed. Return only the JSON specified by response_format.\n\n{guidelines_section}";
            sysPrompt = FillGuidelinesPlaceholder(defaultTemplate);
        }

        var msgs = new System.Collections.Generic.List<Analytics.LLMConversationAnalyzer.ChatMessage>
        {
            new Analytics.LLMConversationAnalyzer.ChatMessage("system", sysPrompt),
            new Analytics.LLMConversationAnalyzer.ChatMessage("user", transcript)
        };

        string raw = null;
        yield return analyzer.RequestChatCompletion(
            messages: msgs.ToArray(),
            maxTokens: 100,
            temperature: 0.0f,
            onDone: v => raw = v,
            debug: _showDebugLogs,
            responseFormat: BuildGuidelinesResponseFormat()
        );

        if (string.IsNullOrWhiteSpace(raw))
        {
            onAllCompleted?.Invoke(AreAllGuidelinesCleared());
            onDone?.Invoke();
            yield break;
        }
        string guidelinesVerdict = null;
        try
        {
            var parsed = Newtonsoft.Json.Linq.JObject.Parse(raw);
            var indicesToken = parsed["completed_indexes"] as Newtonsoft.Json.Linq.JArray;
            if (indicesToken != null)
            {
                var preItems = new System.Collections.Generic.List<string>(activeSection.Items);
                var toRemove = new System.Collections.Generic.List<string>();
                var clearedDetails = new System.Collections.Generic.List<string>();
                foreach (var idxTok in indicesToken)
                {
                    int idx = idxTok.Value<int>();
                    if (idx >= 0 && idx < activeSection.Items.Count)
                    {
                        toRemove.Add(activeSection.Items[idx]);
                        // Use preItems for stable text lookup regardless of subsequent removals
                        var text = (idx >= 0 && idx < preItems.Count) ? preItems[idx] : activeSection.Items[idx];
                        clearedDetails.Add($"{idx}: '{text}'");
                    }
                }
                foreach (var g in toRemove)
                {
                    activeSection.Items.Remove(g);
                }

                if (_showDebugLogs)
                {
                    if (clearedDetails.Count > 0)
                        Debug.Log($"TalkingClip Guidelines Judge: completed {clearedDetails.Count} guideline(s) in section #{_activeSectionIndex}: [" + string.Join(", ", clearedDetails) + "]");
                    else
                        Debug.Log($"TalkingClip Guidelines Judge: no guidelines completed this turn for section #{_activeSectionIndex}.");
                }

                if (clearedDetails.Count > 0)
                {
                    guidelinesVerdict = "Guidelines cleared: " + string.Join(", ", clearedDetails);
                }

                // if section cleared, move to the next
                if (activeSection.Items.Count == 0 && _activeSectionIndex + 1 < _remainingGuidelines.Count)
                {
                    _activeSectionIndex++;
                    if (_showDebugLogs)
                        Debug.Log($"TalkingClip Guidelines Judge: section cleared. Advancing to section #{_activeSectionIndex}.");
                }

#if UNITY_EDITOR
                if (!Application.isPlaying)
                {
                    UnityEditor.EditorUtility.SetDirty(this);
                    UnityEditor.AssetDatabase.SaveAssets();
                }
#endif
            }
        }
        catch (Exception e)
        {
            if (_showDebugLogs) Debug.LogWarning($"Guidelines judge parse failed: {e.Message}; raw: {raw}");
        }
        onAllCompleted?.Invoke(AreAllGuidelinesCleared());
        onVerdict?.Invoke(guidelinesVerdict);
        onDone?.Invoke();
    }

    private bool AreAllGuidelinesCleared()
    {
        EnsureRemainingGuidelinesInitialized();
        if (_remainingGuidelines == null || _remainingGuidelines.Count == 0) return true;
        foreach (var section in _remainingGuidelines)
        {
            if (section?.Items != null && section.Items.Count > 0) return false;
        }
        return true;
    }
    }
}