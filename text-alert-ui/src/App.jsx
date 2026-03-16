import { useState } from "react";
import "./App.css";

export default function App() {
  const [message, setMessage] = useState(
    "בנק לאומי:\nחשבונך נחסם עקב פעילות חריגה.\nלחץ כאן לאימות:\nhttp://faceb00k-login-security.com"
  );

  const [result, setResult] = useState({
    risk_score: 0.91,
    alert_level: "red",
    top_risk: "phishing",
    reasons: [
      "זוהה קישור חשוד שמתחזה לאתר מוכר",
      "ההודעה מבקשת אימות חשבון או מסירת פרטים",
      "יש שימוש בניסוח מלחיץ ודחוף",
    ],
    consequences: [
      "גניבת פרטי התחברות לחשבון",
      "הונאה כספית או השתלטות על החשבון",
    ],
    suspicious_urls: [
      {
        candidate: "http://faceb00k-login-security.com",
        host: "faceb00k-login-security.com",
        score: 92,
        reasons: ["הדומיין דומה למותג מוכר", "שם הדומיין חשוד וכולל החלפת תווים"],
      },
    ],
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const examples = {
    safe: "היי, נתראה מחר ב-20:00 ליד הבית.",
    suspicious:
      "שלום, התקבלה בקשה לאישור פעולה. אם זו לא את, בדקי את החשבון שלך.",
    dangerous:
      "בנק לאומי:\nחשבונך נחסם עקב פעילות חריגה.\nלחץ כאן לאימות:\nhttp://faceb00k-login-security.com",
  };

  const simulate = (kind) => {
    setError("");
    setMessage(examples[kind]);

    if (kind === "safe") {
      setResult({
        risk_score: 0.08,
        alert_level: "green",
        top_risk: "benign",
        reasons: ["לא זוהו סימני הונאה בולטים בהודעה"],
        consequences: ["לא זוהו השלכות חריגות"],
        suspicious_urls: [],
      });
      return;
    }

    if (kind === "suspicious") {
      setResult({
        risk_score: 0.51,
        alert_level: "yellow",
        top_risk: "impersonation",
        reasons: [
          "יש ניסוח שדורש פעולה מצד המשתמש",
          "יש התייחסות לחשבון או לאישור פעולה",
        ],
        consequences: ["מסירת פרטים אישיים", "מעבר לקישור לא בטוח בהמשך"],
        suspicious_urls: [],
      });
      return;
    }

    setResult({
      risk_score: 0.91,
      alert_level: "red",
      top_risk: "phishing",
      reasons: [
        "זוהה קישור חשוד שמתחזה לאתר מוכר",
        "ההודעה מבקשת אימות חשבון או מסירת פרטים",
        "יש שימוש בניסוח מלחיץ ודחוף",
      ],
      consequences: [
        "גניבת פרטי התחברות לחשבון",
        "הונאה כספית או השתלטות על החשבון",
      ],
      suspicious_urls: [
        {
          candidate: "http://faceb00k-login-security.com",
          host: "faceb00k-login-security.com",
          score: 92,
          reasons: ["הדומיין דומה למותג מוכר", "שם הדומיין חשוד וכולל החלפת תווים"],
        },
      ],
    });
  };

  const analyzeMessage = async () => {
    if (!message.trim()) {
      setError("יש להזין הודעה לניתוח.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:8100/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: message }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("לא הצלחתי להתחבר לשרת. ודאי שה-API רץ על פורט 8100 וש-CORS מוגדר.");
    } finally {
      setLoading(false);
    }
  };

  const alertLevel = result?.alert_level || "green";

  const titleMap = {
    red: "⚠️ הודעה מסוכנת",
    yellow: "⚠️ הודעה חשודה",
    green: "✅ הודעה נראית תקינה",
  };

  const subtitleMap = {
    red: "זוהו סימני פישינג ברמה גבוהה",
    yellow: "זוהו סימנים מחשידים שדורשים בדיקה",
    green: "לא זוהו סימני הונאה בולטים",
  };

  const recommendationMap = {
    red: ["לא ללחוץ על הקישור", "לא למסור קוד אימות", "לבדוק מול הגוף הרשמי"],
    yellow: ["להיזהר לפני לחיצה", "לא למסור פרטים אישיים", "לבדוק מול הגוף הרשמי"],
    green: ["אפשר להמשיך בזהירות", "לא למסור קוד אם יתבקש בהמשך"],
  };

  return (
    <div className="app" dir="rtl">
      <div className="layout simple-layout">
        <div className="left-panel compact">
          <div className="input-card">
            <div className="input-header compact-header">
              <div>
                <div className="input-title">בדיקת הודעה</div>
                <div className="input-subtitle">הדביקי כאן הודעה לבדיקה</div>
              </div>

              <div className="example-buttons">
                <button onClick={() => simulate("safe")}>תקינה</button>
                <button onClick={() => simulate("suspicious")}>חשודה</button>
                <button onClick={() => simulate("dangerous")}>פישינג</button>
              </div>
            </div>

            <div className="input-row">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="הדביקי כאן הודעה לבדיקה..."
              />

              <button className="analyze-btn" onClick={analyzeMessage} disabled={loading}>
                {loading ? "מנתח..." : "נתח הודעה"}
              </button>
            </div>

            {error && <div className="error-box">{error}</div>}
          </div>
        </div>

        <div className="phone-wrapper">
          <div className="phone-shell">
            <div className="phone-screen">
              <div className="phone-notch" />

              <div className="phone-header">
                <div className="phone-status">
                  <span>09:41</span>
                  <span>5G 🔋</span>
                </div>
                <div className="phone-title">הודעות</div>
              </div>

              <div className="phone-content">
                <div className="message-bubble">
                  <div className="message-label">הודעה נכנסת</div>
                  <div className="message-text">{message}</div>
                  <div className="message-time">09:40</div>
                </div>

                <div className={`alert-popup ${alertLevel}`}>
                  <div className="alert-icon">
                    {alertLevel === "green" ? "✅" : "⚠️"}
                  </div>

                  <div className="alert-body">
                    <div className="alert-top">
                      <div>
                        <div className="alert-title">{titleMap[alertLevel]}</div>
                        <div className="alert-subtitle">{subtitleMap[alertLevel]}</div>
                      </div>

                      <div className={`risk-badge ${alertLevel}`}>
                        Risk {result?.risk_score}
                      </div>
                    </div>

                    <div className="alert-info-grid">
                      <div className="mini-card">
                        <div className="mini-label">סיכון עיקרי</div>
                        <div className="mini-value">{result?.top_risk}</div>
                      </div>

                      <div className="mini-card">
                        <div className="mini-label">רמת התראה</div>
                        <div className="mini-value">{result?.alert_level}</div>
                      </div>
                    </div>

                    <div className="alert-section">
                      <div className="section-title">למה זה חשוד?</div>
                      <ul>
                        {(result?.reasons || []).slice(0, 3).map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="alert-section">
                      <div className="section-title">מה מומלץ לעשות?</div>
                      <div className="chips">
                        {recommendationMap[alertLevel].map((item) => (
                          <span key={item} className={`chip ${alertLevel}`}>
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>

                    {result?.suspicious_urls?.length > 0 && (
                      <div className="url-box">
                        <div className="section-title">קישור חשוד</div>
                        <div className="url-host">{result.suspicious_urls[0].host}</div>
                        <div className="url-reasons">
                          {(result.suspicious_urls[0].reasons || []).join(" • ")}
                        </div>
                      </div>
                    )}

                    <div className="popup-buttons">
                      <button className="secondary-btn">סגור</button>
                      <button className={`primary-btn ${alertLevel}`}>
                        {alertLevel === "green" ? "אישור" : "אל תלחץ"}
                      </button>
                    </div>
                  </div>
                </div>

                <div className="bottom-spacer" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}