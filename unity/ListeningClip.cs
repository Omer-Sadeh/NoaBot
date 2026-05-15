using System;
using System.Collections;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Events;
using InceptorEngine.Clips.Interfaces;

namespace InceptorEngine.Clips
{
    [CreateAssetMenu(menuName = "Inceptor/Clip/LLM Listening Clip", fileName = "LLMListeningClip", order = 1)]
    [Serializable]
    public class ListeningClip : Clip
    {
    [SerializeField] private int _nextClip = 0;
    
    [Header("Recording Settings")]
    [SerializeField] private bool _showDebugLogs = true;
    [SerializeField] private int _recordingLength = 10;
    [SerializeField] private float _minAmplitude = 0.02f;
    [SerializeField] private int _sampleRate = 24000;

    [Header("VAD Settings")]
    [Tooltip("Seconds of continuous speech needed before we consider speech detected.")]
    [SerializeField] private float _voiceStartSeconds = 0.50f;
    [Tooltip("Seconds of continuous silence after speech to stop listening.")]
    [SerializeField] private float _silenceEndSeconds = 1.5f;
    [Tooltip("Adaptive threshold multiplier over noise floor (RMS).")]
    [SerializeField] private float _noiseMultiplier = 2.5f;
    [Tooltip("Minimum absolute RMS level required, as a floor to avoid ultra-quiet environments sticking.")]
    [SerializeField] private float _minAbsoluteRms = 0.006f;

    // output
    public static string LastUserText;

    private AudioClip _micClip;
    private bool _isRecording;
    private string _micDevice;

        public override void Build(JObject clipData)
        {
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
                if (_showDebugLogs) Debug.LogWarning($"ListeningClip Build: failed to parse next clip metadata: {e.Message}");
            }
        }

        protected override IEnumerator Run(UnityAction<int> onClipEnd)
        {
            var inceptor = Inceptor.GetAvailableInceptors();
            if (inceptor == null)
            {
                Debug.LogError("ListeningClip: Inceptor not found");
                onClipEnd?.Invoke(0);
                yield break;
            }

            if (_showDebugLogs) Debug.Log("ListeningClip: Start mic");

            // start
            _micDevice = Microphone.devices.Length > 0 ? Microphone.devices[0] : null;
            if (_micDevice == null)
            {
                Debug.LogError("ListeningClip: No microphone found");
                onClipEnd?.Invoke(0);
                yield break;
            }
            _micClip = Microphone.Start(_micDevice, false, _recordingLength, _sampleRate);
            _isRecording = true;
            // Wait briefly for the microphone to start producing samples
            float micReadyWait = 0f;
            while (Microphone.GetPosition(_micDevice) <= 0 && micReadyWait < 1.0f)
            {
                yield return new WaitForSeconds(0.05f);
                micReadyWait += 0.05f;
            }
            if (_showDebugLogs) Debug.Log($"ListeningClip: using device '{_micDevice}', requested {_sampleRate}Hz, actual {_micClip.frequency}Hz");

            // VAD wait until speech then silence (adaptive noise floor + RMS + peak)
            float silenceTimer = 0f, voiceTimer = 0f;
            bool voiceDetected = false;
            float elapsed = 0f;
            int lastMicPosition = 0;
            int currentRate = _micClip != null ? _micClip.frequency : _sampleRate;
            // Use ~20ms slices for responsiveness
            int sliceSamples = Mathf.Max(256, currentRate / 50);
            float noiseRms = 0.0f; // adaptive noise floor (RMS)
            bool noiseCalibrated = false;
            bool anyActivity = false; // any above-floor energy seen

            while (_isRecording && !forceStopClip)
            {
                // If underlying recording stopped (length exceeded or device error), bail out
                if (!Microphone.IsRecording(_micDevice))
                {
                    if (_showDebugLogs) Debug.Log("ListeningClip: Microphone stopped recording (buffer length reached or device issue)");
                    break;
                }

                if (_micClip != null)
                {
                    int micPosition = Microphone.GetPosition(_micDevice);
                    if (micPosition >= 0) lastMicPosition = micPosition;

                    // Only analyze once we have enough samples to fill our slice
                    if (micPosition > sliceSamples)
                    {
                        float[] samples = new float[sliceSamples];
                        int start = micPosition - sliceSamples;
                        // Clamp start to valid range
                        if (start < 0) start = 0;
                        _micClip.GetData(samples, start);

                        // Compute RMS and peak
                        double sumSq = 0.0;
                        int len = samples.Length;
                        float peakAbs = 0f;
                        for (int i = 0; i < len; i++)
                        {
                            float v = samples[i];
                            sumSq += (double)v * (double)v;
                            float av = Mathf.Abs(v);
                            if (av > peakAbs) peakAbs = av;
                        }
                        float rms = Mathf.Sqrt((float)(sumSq / len));

                        // Adaptive noise calibration for first ~0.4s or when below current threshold
                        float dt = (float)sliceSamples / currentRate;
                        if (!noiseCalibrated || !voiceDetected)
                        {
                            // Slow ramp to initial noise floor
                            noiseRms = Mathf.Lerp(noiseRms, rms, 0.2f);
                            if (elapsed > 0.4f) noiseCalibrated = true;
                        }
                        else if (!voiceDetected && rms < noiseRms * 1.2f) // update floor only when near/below it and before voice
                        {
                            noiseRms = Mathf.Lerp(noiseRms, rms, 0.05f);
                        }

                        float dynamicThreshold = Mathf.Max(_minAbsoluteRms, noiseRms * Mathf.Max(1.2f, _noiseMultiplier));
                        float peakThreshold = Mathf.Max(_minAmplitude * 1.5f, dynamicThreshold * 1.2f);
                        bool isSpeech = (rms > Mathf.Max(_minAmplitude, dynamicThreshold)) || (peakAbs > peakThreshold);

                        if (isSpeech)
                        {
                            voiceTimer += dt;
                            silenceTimer = 0f;
                            anyActivity = true;
                            if (!voiceDetected && voiceTimer >= _voiceStartSeconds)
                            {
                                voiceDetected = true;
                                if (_showDebugLogs) Debug.Log($"ListeningClip: voice detected (rms={rms:F3}, peak={peakAbs:F3}, floor={noiseRms:F3}, thr={dynamicThreshold:F3})");
                            }
                            // Fast-path: very strong transient speech can start immediately
                            if (!voiceDetected && peakAbs > _minAmplitude * 3f)
                            {
                                voiceDetected = true;
                                if (_showDebugLogs) Debug.Log($"ListeningClip: voice detected (strong peak {peakAbs:F3})");
                            }
                        }
                        else
                        {
                            silenceTimer += dt;
                            // small decay to avoid sticking voiceTimer high forever
                            voiceTimer = Mathf.Max(0f, voiceTimer - dt * 0.5f);
                            if (voiceDetected && silenceTimer >= _silenceEndSeconds)
                            {
                                if (_showDebugLogs) Debug.Log($"ListeningClip: silence detected, stopping (rms={rms:F3}, peak={peakAbs:F3}, floor={noiseRms:F3}, thr={dynamicThreshold:F3})");
                                break;
                            }
                            // Fallback: if we saw some activity but never fully reached voice start, still stop after some quiet period
                            if (!voiceDetected && anyActivity && silenceTimer >= Mathf.Max(0.6f, _silenceEndSeconds))
                            {
                                if (_showDebugLogs) Debug.Log("ListeningClip: brief speech then silence, stopping");
                                break;
                            }
                        }

                        elapsed += dt;
                    }
                }
                // Sleep a short while to avoid tight loop; we base timers on sample counts
                yield return new WaitForSeconds(0.02f);
                // Hard cap to avoid getting stuck when no speech is detected
                if (elapsed >= _recordingLength + 0.5f)
                {
                    if (_showDebugLogs) Debug.Log("ListeningClip: max listen time reached without complete VAD, stopping");
                    break;
                }
            }

            // stop
            int finalPosition = 0;
            if (_isRecording)
            {
                // Capture final mic position before stopping; if we couldn't read one, use last seen
                finalPosition = Microphone.IsRecording(_micDevice) ? Microphone.GetPosition(_micDevice) : lastMicPosition;
                if (finalPosition < 0) finalPosition = 0;
                Microphone.End(_micDevice);
                _isRecording = false;
            }

            // hand over to analyzer for STT
            string resultText = null;
            if (_micClip != null)
            {
                AudioClip toTranscribe = _micClip;
                // Trim to actual recorded length if we have a reasonable final position
                if (finalPosition > 0 && finalPosition < _micClip.samples)
                {
                    int channels = _micClip.channels;
                    int trimSamples = finalPosition * channels;
                    var data = new float[trimSamples];
                    try { _micClip.GetData(data, 0); }
                    catch { /* fallback to untrimmed if GetData fails */ data = null; }
                    if (data != null)
                    {
                        var trimmed = AudioClip.Create("ListeningClip_Recorded", finalPosition, channels, _micClip.frequency, false);
                        trimmed.SetData(data, 0);
                        toTranscribe = trimmed;
                    }
                }

                var analyzer = FindOrCreateAnalyzer(inceptor);
                byte[] wav = Analytics.LLMConversationAnalyzer.ClipToWav(toTranscribe);
                yield return inceptor.StartCoroutine(analyzer.TranscribeWav(wav, t => resultText = t, _showDebugLogs));
            }

            LastUserText = resultText;
            if (_showDebugLogs) Debug.Log($"ListeningClip USER: {LastUserText}");

            if (_endDelay != 0) yield return new WaitForSeconds(_endDelay);
            int next = _nextClip;
            onClipEnd?.Invoke(next);
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
        private static void InitializeStatics() { LastUserText = null; }
    }
}
