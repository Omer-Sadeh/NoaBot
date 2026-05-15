# Unity WebGL → Firestore + Firebase Hosting Guide

A practical setup for collecting gameplay statistics from a Unity WebGL game into Firestore (using the REST API, no Firebase SDK needed), and hosting the game on Firebase.

This was written for a research-study setup where participants play a short game and are then redirected to an external questionnaire.

---

## Why the REST API instead of the Firebase Unity SDK?

The official Firebase Unity SDK does not support WebGL builds. The two workarounds are:

1. **Firestore REST API** (what this guide uses) — call the public Firestore HTTP endpoint with `UnityWebRequest`. No SDK, no `index.html` edits, works in editor and in WebGL builds identically.
2. **JavaScript bridge** — load the Firebase JS SDK in `index.html` and call it from Unity via a `.jslib` plugin.

For write-only telemetry the REST API is simpler and just as good.

---

## Part 1 — Set up Firestore

### 1.1 Create the project (skip if you already have one)

1. Go to [console.firebase.google.com](https://console.firebase.google.com) and create a new project (or reuse the one from your Streamlit setup).
2. In the left sidebar, open **Build → Firestore Database** and click **Create database**. Pick a location close to your participants. Start in **production mode** (we'll write proper rules below).

### 1.2 Get your project ID and Web API key

1. Click the gear icon → **Project settings**.
2. Note the **Project ID** — for this project it is `noabotprompts`.
3. Under **Your apps**, click the `</>` (Web) icon to register a web app if you haven't already. Give it any nickname; you do **not** need to set up Firebase Hosting at this prompt.
4. Copy the **Web API Key** from the config snippet (the `apiKey` field).

You'll paste both of these into your Unity script.

> The Web API key is not a secret — it's safe to ship in the build. Access control is done via security rules, not by hiding the key.

### 1.3 Write security rules

In the Firestore console → **Rules** tab. Replace with something like:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    // Session parent document — written by both Streamlit and Unity.
    match /sessions/{sessionId} {
      allow create, update: if
        request.resource.data.mode     is string
        && request.resource.data.status   is string
        && request.resource.data.language is string;

      allow read, delete: if false;

      // Conversation sub-documents written at game end.
      match /conversations/{docId} {
        allow create, update: if
          request.resource.data.keys().hasOnly([
            'timestamp', 'data', 'mode', 'status',
            'is_successful', 'session_finished',
            'user_message_count', 'completed_guidelines', 'current_stage'
          ])
          && request.resource.data.data   is string
          && request.resource.data.mode   is string
          && request.resource.data.status is string;

        allow read, delete: if false;
      }
    }
  }
}
```

This mirrors the schema the Streamlit app uses. The parent document holds session metadata; the `conversations` sub-collection holds the transcript and stats. Click **Publish**.

When you want to download the data later, use a service-account key from Python or the Firebase console (admin SDK bypasses these rules).

---

## Part 2 — Send data from Unity

### 2.1 The Firestore document format (one quirk to know)

The REST API doesn't accept plain JSON. Each field has to be wrapped with its type:

```json
{
  "fields": {
    "mode":     { "stringValue": "open" },
    "status":   { "stringValue": "completed" },
    "language": { "stringValue": "en" },
    "created":  { "timestampValue": "2026-05-15T14:00:00Z" }
  }
}
```

Common types: `stringValue`, `integerValue` (as string), `doubleValue`, `booleanValue`, `timestampValue` (ISO 8601, UTC), `arrayValue`, `mapValue`, `nullValue`.

The app uses **two writes** per session — one PATCH to `sessions/{sessionId}` for metadata, and one PATCH to `sessions/{sessionId}/conversations/final_{epoch}` for the transcript and stats. Both use the HTTP `PATCH` method so the call is idempotent (safe to retry).

### 2.2 The C# script

Drop this on a GameObject (e.g. one called `FirestoreClient`) somewhere in your scene. The project ID is already set to match this project; paste your Web API key in the Inspector.

```csharp
using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class FirestoreClient : MonoBehaviour
{
    [Header("Firebase")]
    [SerializeField] private string projectId = "noabotprompts";
    [SerializeField] private string apiKey    = "your-web-api-key";

    private const string BaseUrl = "https://firestore.googleapis.com/v1/projects";

    /// <summary>Call this when the game ends.</summary>
    public void SendSessionData(
        string sessionId,
        string mode,               // "open" or "closed"
        string language,           // "en" or "he"
        bool   isSuccessful,
        bool   sessionFinished,
        int    userMessageCount,
        int    completedGuidelines,
        int    currentStage,
        string transcript)
    {
        StartCoroutine(PostSession(sessionId, mode, language, isSuccessful,
            sessionFinished, userMessageCount, completedGuidelines, currentStage, transcript));
    }

    private IEnumerator PostSession(
        string sessionId, string mode, string language,
        bool isSuccessful, bool sessionFinished,
        int userMessageCount, int completedGuidelines, int currentStage,
        string transcript)
    {
        string now  = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ");
        string root = $"{BaseUrl}/{projectId}/databases/(default)/documents/sessions";

        // 1. Write parent session document (metadata).
        string parentUrl  = $"{root}/{sessionId}?key={apiKey}";
        string parentJson =
            "{\"fields\":{" +
            $"\"created\":{{\"timestampValue\":\"{now}\"}}," +
            $"\"last_updated\":{{\"timestampValue\":\"{now}\"}}," +
            $"\"mode\":{{\"stringValue\":\"{Escape(mode)}\"}}," +
            $"\"status\":{{\"stringValue\":\"completed\"}}," +
            $"\"language\":{{\"stringValue\":\"{Escape(language)}\"}}" +
            "}}";

        yield return PatchDocument(parentUrl, parentJson);

        // 2. Write conversation sub-document (transcript + stats).
        long   epoch   = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string convUrl = $"{root}/{sessionId}/conversations/final_{epoch}?key={apiKey}";
        string convJson =
            "{\"fields\":{" +
            $"\"timestamp\":{{\"timestampValue\":\"{now}\"}}," +
            $"\"data\":{{\"stringValue\":\"{Escape(transcript)}\"}}," +
            $"\"mode\":{{\"stringValue\":\"{Escape(mode)}\"}}," +
            $"\"status\":{{\"stringValue\":\"completed\"}}," +
            $"\"is_successful\":{{\"booleanValue\":{(isSuccessful ? "true" : "false")}}}," +
            $"\"session_finished\":{{\"booleanValue\":{(sessionFinished ? "true" : "false")}}}," +
            $"\"user_message_count\":{{\"integerValue\":\"{userMessageCount}\"}}," +
            $"\"completed_guidelines\":{{\"integerValue\":\"{completedGuidelines}\"}}," +
            $"\"current_stage\":{{\"integerValue\":\"{currentStage}\"}}" +
            "}}";

        yield return PatchDocument(convUrl, convJson);

        Debug.Log("Session saved.");
    }

    private IEnumerator PatchDocument(string url, string json)
    {
        using (UnityWebRequest req = new UnityWebRequest(url, "PATCH"))
        {
            byte[] body = Encoding.UTF8.GetBytes(json);
            req.uploadHandler   = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
                Debug.LogError($"Firestore PATCH failed: {req.error}\n{req.downloadHandler.text}");
        }
    }

    // Minimal JSON string escaping for user-provided strings.
    private static string Escape(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        var sb = new StringBuilder(s.Length);
        foreach (char c in s)
        {
            switch (c)
            {
                case '\"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\b': sb.Append("\\b");  break;
                case '\f': sb.Append("\\f");  break;
                case '\n': sb.Append("\\n");  break;
                case '\r': sb.Append("\\r");  break;
                case '\t': sb.Append("\\t");  break;
                default:
                    if (c < 0x20) sb.AppendFormat("\\u{0:x4}", (int)c);
                    else          sb.Append(c);
                    break;
            }
        }
        return sb.ToString();
    }
}
```

### 2.3 Calling it at game-end

From whatever script ends your game:

```csharp
using System;
using System.Collections;
using System.Text;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    [SerializeField] private FirestoreClient firestore;
    [SerializeField] private string questionnaireUrl = "https://www.surveymonkey.com/r/NP7M559";

    private string sessionId;
    private int    completedGuidelines;
    private int    currentStage;
    private int    userMessageCount;
    private readonly StringBuilder transcript = new StringBuilder();

    void Start()
    {
        sessionId = Guid.NewGuid().ToString("N");
    }

    // Call as each message is exchanged during the session.
    public void RecordMessage(string speaker, string text)
    {
        transcript.AppendLine($"-- {speaker}: {text}");
        if (speaker != "Noa") userMessageCount++;
    }

    public void OnGuidelineCompleted() => completedGuidelines++;
    public void OnStageAdvanced()      => currentStage++;

    public void OnGameComplete(bool allGuidelinesCompleted)
    {
        firestore.SendSessionData(
            sessionId:           sessionId,
            mode:                "open",
            language:            "en",
            isSuccessful:        allGuidelinesCompleted,
            sessionFinished:     true,
            userMessageCount:    userMessageCount,
            completedGuidelines: completedGuidelines,
            currentStage:        currentStage,
            transcript:          transcript.ToString()
        );

        StartCoroutine(RedirectAfterDelay(1.5f));
    }

    private IEnumerator RedirectAfterDelay(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        // Append session ID so you can join game data with questionnaire responses.
        Application.OpenURL($"{questionnaireUrl}?pid={sessionId}");
    }
}
```

The `sessionId` flows into the questionnaire URL as `?pid=…`. SurveyMonkey, Qualtrics, and most other tools can capture that into a hidden field so you can join the two datasets later.

### 2.4 Test before building

Press Play in the editor and call `OnGameComplete(true)`. Watch the **Console** for "Session saved." and check the Firestore console — a new document should appear under `sessions/{sessionId}`, and a `conversations/final_…` sub-document should appear inside it. If you get a 403, the security rules are rejecting the payload; a mismatch between the rule's field list and what you're sending is the most common cause.

### 2.5 Testing with curl

You can verify the endpoint and schema without touching Unity at all. Run these two commands from your terminal (replace `YOUR_API_KEY`):

```bash
PROJECT="noabotprompts"
API_KEY="YOUR_API_KEY"
SESSION="test-$(date +%s)"
BASE="https://firestore.googleapis.com/v1/projects/$PROJECT/databases/(default)/documents/sessions"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Create / update the parent session document.
curl -s -X PATCH \
  "$BASE/$SESSION?key=$API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"fields\": {
      \"created\":      { \"timestampValue\": \"$NOW\" },
      \"last_updated\": { \"timestampValue\": \"$NOW\" },
      \"mode\":         { \"stringValue\": \"open\" },
      \"status\":       { \"stringValue\": \"completed\" },
      \"language\":     { \"stringValue\": \"en\" }
    }
  }" | python3 -m json.tool

# 2. Write the conversation sub-document.
EPOCH="$(date +%s)"
curl -s -X PATCH \
  "$BASE/$SESSION/conversations/final_$EPOCH?key=$API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"fields\": {
      \"timestamp\":           { \"timestampValue\": \"$NOW\" },
      \"data\":                { \"stringValue\": \"-- Noa: Hello\\n\\n-- User: Hi Noa\\n\" },
      \"mode\":                { \"stringValue\": \"open\" },
      \"status\":              { \"stringValue\": \"completed\" },
      \"is_successful\":       { \"booleanValue\": true },
      \"session_finished\":    { \"booleanValue\": true },
      \"user_message_count\":  { \"integerValue\": \"5\" },
      \"completed_guidelines\": { \"integerValue\": \"3\" },
      \"current_stage\":       { \"integerValue\": \"2\" }
    }
  }" | python3 -m json.tool
```

A successful write returns the full document JSON. A 403 means the security rules blocked it — check the field names match exactly.

---

## Part 3 — Build for WebGL

1. **File → Build Settings → WebGL → Switch Platform** (if not already).
2. **Player Settings → Publishing Settings → Compression Format → Disabled.**
   Unity's built-in Brotli/Gzip compression breaks on Firebase Hosting unless you write a matching `firebase.json`. The cleanest fix is to disable Unity's compression and let Firebase compress on the fly — no quality loss.
3. (Optional) Set a fixed canvas size in **Player Settings → Resolution and Presentation** if you want a predictable layout.
4. Click **Build** and pick an output folder, e.g. `MyGameBuild/`. You should end up with `index.html`, a `Build/` folder, a `TemplateData/` folder, and a `StreamingAssets/` folder if you have any.

---

## Part 4 — Host on Firebase

### 4.1 Install the CLI (one time)

Requires Node.js. Then:

```bash
npm install -g firebase-tools
firebase login
```

### 4.2 Initialize hosting

From inside your build folder:

```bash
cd /path/to/MyGameBuild
firebase init hosting
```

Answer the prompts:

| Prompt | Answer |
|---|---|
| Use an existing project | yes — pick the same project as your Firestore |
| Public directory | `.` (current folder) |
| Single-page app? | **No** |
| Set up automatic builds with GitHub? | No |
| Overwrite `index.html`? | **No** — keep Unity's |

This creates `firebase.json` and `.firebaserc` in the folder.

### 4.3 Deploy

```bash
firebase deploy --only hosting
```

Output looks like:

```
Hosting URL: https://your-project.web.app
```

That's the URL you send participants. It's HTTPS by default, so calls to Firestore from the game won't trigger mixed-content blocks.

### 4.4 Iterating

After every Unity rebuild, just rerun:

```bash
firebase deploy --only hosting
```

Same URL each time. Hard-refresh in the browser (Ctrl+Shift+R / Cmd+Shift+R) if a participant somehow loads a cached version during testing.

---

## Part 5 — Quick checklist before going live

- [ ] Security rules block reads, updates, and deletes from clients.
- [ ] Field names and types in your Unity script match the rule schema exactly.
- [ ] Unity compression is set to **Disabled**.
- [ ] You tested the full loop end-to-end on the deployed URL — not just in the editor.
- [ ] The questionnaire URL is correct and the participant ID flows through if you need it.
- [ ] You've checked the [Firebase free-tier quotas](https://firebase.google.com/pricing) against your expected participant count. The Spark plan gives you 50K Firestore writes/day and 360 MB/day Hosting bandwidth, which is generous for a study but not infinite.

---

## Troubleshooting

**`PERMISSION_DENIED: Missing or insufficient permissions.`** — Security rules rejected the write. Check that the field names, the field types, and the `keys().hasOnly([...])` list all match what your script is sending.

**`HTTP/1.1 400 Bad Request` with "Invalid JSON payload".** — Your JSON is malformed. Most often a string field contains an unescaped quote, newline, or backslash that you didn't run through `Escape()`. Or a `doubleValue` got serialized with a comma decimal separator (use `CultureInfo.InvariantCulture`).

**WebGL build loads forever or fails with a decompression error on Firebase.** — Unity's compression is on. Set Compression Format to Disabled and rebuild.

**`Application.OpenURL` does nothing in WebGL.** — Browsers block popups not triggered by a user gesture. Make sure the call happens inside a button-click handler, not from `Start()` or a timer that fires without interaction.

**CORS error in the browser console when posting to Firestore.** — Firestore's REST endpoints have CORS enabled, so this usually means you typoed the URL (the project ID, the `(default)` literal database name, or the collection path).
