using System;
using System.Collections;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Events;
using TMPro;
using InceptorEngine.Clips.Interfaces;

namespace InceptorEngine.Clips
{
    [CreateAssetMenu(menuName = "Inceptor/Clip/Simulation End Clip", fileName = "SimulationEndClip", order = 1)]
    [Serializable]
    public class SimulationEndClip : Clip
    {
        [Header("Routing")]
        [SerializeField] private int _nextClip = -1;

        [Header("UI")]
        [SerializeField] private bool _showDebugLogs = true;
        [SerializeField] private string _verdictTextTag = "VerdictText";
        [TextArea] [SerializeField] private string _successMessage = "Simulation complete.";
        [SerializeField] private bool _requireContinue = true;

        private TMP_Text _verdictTMP;
        private bool _hasUserClickedContinue;

        public override void Build(JObject clipData)
        {
            Initialize(clipData);
            try
            {
                var meta = clipData?["metadata"];
                if (meta != null && meta["nextClip"] != null)
                {
                    _nextClip = meta["nextClip"].Value<int>();
                }
            }
            catch (Exception e)
            {
                if (_showDebugLogs) Debug.LogWarning($"SimulationEndClip Build: failed to parse next clip metadata: {e.Message}");
            }
        }

        protected override IEnumerator Run(UnityAction<int> onClipEnd)
        {
            CacheVerdictTMP();
            PrintVerdict(_successMessage);
            SetVerdictVisible(true);

            SetContinueButtonActive(_requireContinue);
            if (_requireContinue)
            {
                _hasUserClickedContinue = false;
                while (!_hasUserClickedContinue) { yield return null; }
            }
            else if (_endDelay != 0)
            {
                yield return new WaitForSeconds(_endDelay);
            }

            SetContinueButtonActive(false);
            onClipEnd?.Invoke(_nextClip);
        }

        public void ProceedToNextClip() { _hasUserClickedContinue = true; }

        private void PrintVerdict(string verdict)
        {
            CacheVerdictTMP();

            if (_verdictTMP != null)
                _verdictTMP.text = verdict;
            else if (_showDebugLogs)
                Debug.LogWarning($"SimulationEndClip: No TMP_Text found with tag {_verdictTextTag}");
        }

        private void SetContinueButtonActive(bool active)
        {
            var btn = FindContinueButton();
            if (btn != null)
            {
                var parent = btn.transform != null ? btn.transform.parent : null;
                var container = parent != null ? parent.gameObject : btn;
                container.SetActive(active);
            }
            else if (_showDebugLogs)
            {
                Debug.LogWarning("SimulationEndClip: No GameObject found with tag 'ContinueButton'");
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
                Debug.LogWarning($"SimulationEndClip: No GameObject found with tag {_verdictTextTag}");
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
    }
}
