PROJECT="noabotprompts"
API_KEY="AIzaSyBtSnD_Y-050ODJex-v2gkoSvFzxGLWO6k"
SESSION="test-$(date +%s)"
BASE="https://firestore.googleapis.com/v1/projects/$PROJECT/databases/(default)/documents/sessions_test"
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