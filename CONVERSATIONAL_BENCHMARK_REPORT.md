# Conversational Speech & Dialect Benchmark Report

## Evaluation of Realistic, Spoken-Style Inquiries Across Chhattisgarhi, Hindi, and English

**Execution Date:** 2026-11-09 23:05:35\
**Evaluation Mode:** Realistic Conversational Utterances (Greetings, Colloquial Dialects, Slang, Scenario Questions)\
**Total Questions Evaluated:** 27 (9 Chhattisgarhi, 9 Hindi, 9 English)\
**Target Environment:** Local GPU Workstation (RTX 3050 Laptop GPU, Faster-Whisper STT, Coqui/SAPI TTS)\
**Inter-Query Pacing:** 4.5s (Rate-Limit Guarding)\
**Overall Conversational Success Rate:** **92.6%** (25/27 Passed / Safely Refused)

---

## 1. Executive Summary & Scorecard

| Conversational Metric                   | Chhattisgarhi (छत्तीसगढ़ी) | Hindi (हिन्दी)         | English             | Overall System |
| --------------------------------------- | -------------------------- | ---------------------- | ------------------- | -------------- |
| **Total Questions**                     | 9                          | 9                      | 9                   | **27**         |
| **Conversational Pass Rate**            | **88.9%**                  | **88.9%**              | **100.0%**          | **92.6%**      |
| **Direct Accurate Passes**              | 7                          | 8                      | 8                   | **23**         |
| **Safe Refusals (Chitchat/OOS)**        | 1                          | 0                      | 1                   | **2**          |
| **STT Transcription Success**           | 100%                       | 100%                   | 100%                | **100%**       |
| **TTS Audio Synthesis Success**         | 100%                       | 100%                   | 100%                | **100%**       |
| **TTS Voice Configuration**             | Coqui VITS (1.25X)         | Coqui VITS (1.25X)     | Windows SAPI (1.0X) | Hybrid         |
| **Devanagari Numbers Normalization**    | Enabled (_पंद्रह हज़ार_)   | Enabled (_दस बजे रात_) | Standard Words      | Compliant      |
| **Avg Retrieval Latency**               | 9.673s                     | 3.28s                  | 2.414s              | **5.123s**     |
| **Avg LLM Generation Latency**          | 1.489s                     | 1.704s                 | 1.348s              | **1.514s**     |
| **Avg TTS Latency (first 3 sentences)** | 741.5ms                    | 729.4ms                | 119.4ms             | **530.1ms**    |
| **Avg Total Latency**                   | 12.271s                    | 7.832s                 | 5.851s              | **8.651s**     |

---

## 2. Detailed Conversational Transcript & Analysis

### [CG-CONV-01] B.Tech Admissions Process (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "जय जोहार संगी, मोर एक ठन बात सुन ना, बीटेक म एडमिशन बर का का करना परही?"
- **Effective Query (Rewriter):** `What is the admission procedure and eligibility criteria for B.Tech programs at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (B. Tech. Admissions 2026:-  Application for Spot Round) (Contexts: 4)
- **Latencies:** Retrieval: `4.309s` | LLM: `1.771s` | TTS: `1151.7ms` | Total: `7.58s`
- **System Spoken Response:**

> जय जोहार!
>
> आईआईआईटी नया रायपुर म बीटेक प्रोग्राम म प्रवेश बर उम्मीदवार ल जियांग मुख्य परीक्षा (जेईई मेन) के प्रदर्शन के आधार पर मेरिट के अनुसार प्रवेश मिलथे। छत्तीसगढ़ कोटा बर उम्मीदवार ल छत्तीसगढ़ राज्य के स्कूल ले दसवां अउ बारहवां के परीक्षा पास करना जररी हे, अउ बारहवां के परीक्षा म रसायन शास्त्र, भौतिकी अउ गणित विषय म कुल पचास प्रतिशत अंक (अनुसूचित जाति, अनुसूचित जनजाति, अन्य पिछड़ा वर्ग अउ दिव्यांग उम्मीदवार बर पैंतालीस प्रतिशत) होना चाहिए।
>
> छत्तीसगढ़ कोटा अउ एनटीपीसी कोटा बर उम्मीदवार मन ल संस्थान के ऑनलाइन पोर्टल (https://www.iiitnr.ac.in/content/b-tech-admission-2026) के माध्यम ले आवेदन करना पड़ही अउ पांच सौ रुपिया आवेदन शुल्क ऑनलाइन जमा करना पड़ही। ऑल इंडिया कोटा के सीट जोसा काउंसलिंग के माध्यम ले भरे जाथे।
>
> प्रवेश के समय देना पड़े वाला मुख्य दस्तावेज ए प्रकार हे:
>
> - दसवां के पास मार्कशीट
> - बारहवां के पास मार्कशीट
> - मूल ट्रांसफर सर्टिफिकेट
> - मूल कैरेक्टर सर्टिफिकेट
> - मूल माइग्रेशन सर्टिफिकेट
> - जियांग मुख्य परीक्षा रैंक कार्ड
> - दिव्यांगता प्रमाण पत्र (यदि लागू होय)
>
> 📄 Source: Page 12 — eligibility will be as per the business rules of JoSAA/CSAB 2026. All

---

### [CG-CONV-02] Chhattisgarh State Quota (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "हमर छत्तीसगढ़ के लइकामन बर बीटेक म कोनो कोटा हे का संगी? कतेक प्रतिशत सीट आरक्षित हे?"
- **Effective Query (Rewriter):** `What is the state quota reservation percentage and eligibility criteria for Chhattisgarh domicile students in B.Tech admissions at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (B. Tech. Admissions 2026:-  Application for Spot Round) (Contexts: 4)
- **Latencies:** Retrieval: `3.32s` | LLM: `1.554s` | TTS: `793.8ms` | Total: `6.06s`
- **System Spoken Response:**

> जय जोहार!
>
> हाँ संगी, हमर छत्तीसगढ़ के लइकामन बर बीटेक प्रोग्राम म कोटा हावे। छत्तीसगढ़ कोटा के अंतर्गत पचास प्रतिशत सीट आरक्षित हे, जेकर मतलब एक सौ पैंतीस सीट (कंप्यूटर साइंस म पैंतालीस, इलेक्ट्रॉनिकी म पैंतालीस, अउ डेटा साइंस म पैंतालीस सीट) ए श्रेणी म आरक्षित हावे। ए कोटा के अंतर्गत सब रिजर्वेशन छत्तीसगढ़ शासन के नियम के हिसाब से होही।
>
> 📄 Source: Page 9 — eligibility will be as per the business rules of JoSAA/CSAB 2026. All

---

### [CG-CONV-03] Hostel Life & Mess Charges (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "हॉस्टल म रहे के अउ खाय-पिये के मेस चार्ज कतेक रुपिया लगथे हर सेमेस्टर म?"
- **Effective Query (Rewriter):** `What is the per-semester fee for hostel accommodation and mess charges at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 6 (M.Tech (DSAI) with CMIT Fellowship 2026) (Contexts: 4)
- **Latencies:** Retrieval: `3.232s` | LLM: `1.32s` | TTS: `696.0ms` | Total: `5.79s`
- **System Spoken Response:**

> जय जोहार!
>
> आईआईआईटी नया रायपुर के दस्तावेज के अनुसार, एम.टेक (डी.एस.ए.आई.) विथ सीएमआईटी फेलोशिप दो हज़ार छब्बीस कार्यक्रम बर हर सेमेस्टर म हॉस्टल फीस पंद्रह हज़ार रुपिया अउ मेस एडवांस पच्चीस हज़ार रुपिया लगथे।
>
> 📄 Source: Page 6 — M.Tech (DSAI) with CMIT Fellowship 2026

---

### [CG-CONV-04] Branch Sliding Criteria (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "संगी, मोर ब्रांच ल बदल के सीएसई करना हे फर्स्ट ईयर के बाद, ओखर बर कतका सीपीआई लाना परही?"
- **Effective Query (Rewriter):** `What is the minimum CPI requirement for branch change to Computer Science and Engineering (CSE) after the first year at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 6 (Section 5: Branch Sliding & Branch Change Regulations) (Contexts: 1)
- **Latencies:** Retrieval: `3.78s` | LLM: `1.793s` | TTS: `804.2ms` | Total: `6.75s`
- **System Spoken Response:**

> जय जोहार!
>
> अपन ब्रांच ल बदल के कंप्यूटर साइंस एंड इंजीनियरिंग (CSE) करे बर, पहिली एकेडमिक बछर (दूसरा सेमेस्टर के अंत म) तोर कम से कम आठ दशमलव शून्य (आठ) सीपीआई (CPI) होना अनिवार्य हे। एकर साथे-साथ तोर पहिली अउ दूसर सेमेस्टर के सबो कोर्स पहिली ही प्रयास म पास होना चाही, मने कि कोनो एक्टिव बैकलॉग या एफ (F) ग्रेड नइ होना चाही।
>
> 📄 Source: Page 6 — Section 5: Branch Sliding & Branch Change Regulations

---

### [CG-CONV-05] Medical Attendance Relaxation (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "अगर मोर तबीयत खराब होगे अउ क्लास नइ जा सकेंव, त उपस्थिति म कोनो छूट मिलही का?"
- **Effective Query (Rewriter):** `What is the attendance policy and exemption rule for medical leave or illness at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (Section 2: Attendance Regulations & Medical Condonation) (Contexts: 3)
- **Latencies:** Retrieval: `2.909s` | LLM: `1.539s` | TTS: `896.7ms` | Total: `5.59s`
- **System Spoken Response:**

> जय जोहार!
>
> हावे, अगर तोर तबियत खराब होथे त उपस्थिति (अटेंडेंस) म छूट मिल सकथे। नियम के अनुसार, प्रमाणित चिकित्सीय आधार (जैसे गंभीर बीमारी अउ अस्पताल म भर्ती होना) बर सक्षम प्राधिकारी (शैक्षणिक मामलों के डीन) द्वारा उपस्थिति म अधिकतम दस प्रतिशत तक के छूट (यानी न्यूनतम पचहत्तर प्रतिशत ले घटा के न्यूनतम पैंसठ प्रतिशत तक) दे जा सकथे।
>
> एखर बर तोर या तोर पालक ला कैंपस म लउटे के सात दिन के भीतर वैध मेडिकल सर्टिफिकेट अउ अस्पताल के डिस्चार्ज समरी के संग एकेडमिक अफेयर्स ऑफिस म एक झन औपचारिक मेडिकल लीव आवेदन जमा करै बर पड़ही।
>
> 📄 Source: Page 3 — Section 2: Attendance Regulations & Medical Condonation

---

### [CG-CONV-06] Passing Marks & Minimum Grade (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "अरे भाई, विषय पास करे बर कम से कम कोन सा ग्रेड चाहिए संस्थान म?"
- **Effective Query (Rewriter):** `What is the minimum passing grade required in a course to pass academic subjects at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 4 (Section 3: 10-Point Letter Grading Scale & Evaluation) (Contexts: 1)
- **Latencies:** Retrieval: `2.928s` | LLM: `1.95s` | TTS: `642.2ms` | Total: `5.74s`
- **System Spoken Response:**

> जय जोहार!
>
> आईआईआईटी नया रायपुर म कोई भी विषय ला पास करै बर कम से कम डी (D) ग्रेड (चार ग्रेड पाइंट) की जरूरत पड़थ हे। एखर से कम ग्रेड यानी एफ (F) मिलने म विषय पास नइ होवय, अउ छात्र ला फेर से रजिस्ट्रेशन करा के या संपूरक परीक्षा दे के ओ विषय ला क्लियर करै बर पड़थ हे।
>
> 📄 Source: Page 4 — Section 3: 10-Point Letter Grading Scale & Evaluation

---

### [CG-CONV-07] Hostel Night Curfew (Chhattisgarhi) — ⚠️ GUARD_FLAG (missing_page_citation)

- **User Utterance (Spoken):** "संझा के हॉस्टल कतका बेरा तक वापस आ सकथन? रात म कतका बेरा गेट बंद हो जाथे?"
- **Effective Query (Rewriter):** `What are the hostel out-time and night entry gate closing timings for students at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 4 (Hostel Living & Night Gate Regulations) (Contexts: 2)
- **Latencies:** Retrieval: `3.129s` | LLM: `1.218s` | TTS: `548.0ms` | Total: `5.28s`
- **System Spoken Response:**

> जय जोहार!
>
> आईआईआईटी नया रायपुर म रहइया सबो छात्र-छात्रा मन बर हॉस्टल आवे के अनिवार्य समय रात के दस बज तक हावे। छात्र मन ल संझा या रात के दस बज तक हॉस्टल परिसर म अनिवार्य रूप ले वापस आ जाना चाही।
>
> 📄 Source: Page चार — Hostel Living & Night Gate Regulations

---

### [CG-CONV-08] Health Center & Ambulance (Chhattisgarhi) — ✅ PASS

- **User Utterance (Spoken):** "परिसर म अगर अचानक कोनो के तबीयत खराब होगे त डॉक्टर या एम्बुलेंस के का सुभीता हे?"
- **Effective Query (Rewriter):** `What medical facilities, campus doctor availability, and emergency ambulance services are available on the IIIT-NR campus for sudden illness?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (Medical Center, Dispensary & 24x7 Ambulance) (Contexts: 2)
- **Latencies:** Retrieval: `59.967s` | LLM: `2.257s` | TTS: `640.8ms` | Total: `63.25s`
- **System Spoken Response:**

> जय जोहार!
>
> आईआईआईटी नया रायपुर के परिसर म अगर अचानक कोनो के तबीयत खराब हो जाथे, त छात्र मन बर स्वास्थ्य सुविधा अउ एम्बुलेंस के बहुत अच्छा सुभीता हावे। एखर विवरण नीचे लिखे मुताबिक हे:
>
> - **परिसर म स्वास्थ्य केंद्र (डिस्पेंसर):** संस्थान म एक समर्पित ऑन-कैंपस स्वास्थ्य केंद्र हावे, जिहाँ एक निवासी चिकित्सा अधिकारी (डॉक्टर), प्रशिक्षित नर्स अउ आपातकालीन चिकित्सा परिचारक (अटेंडेंट) मौजूद रहिथें। इहाँ प्राथमिक उपचार, नियमित बाह्य रोगी (ओपीडी) परामर्श, बुनियादी नैदानिक जाँच अउ जरूरी दवाई मिलथें।
> - **चौबीसों घंटा एम्बुलेंस सुभीता:** एक समर्पित अउ पूरी तरह लैस आपातकालीन एम्बुलेंस सातो दिन चौबीसों घंटा परिसर म तैनात रहिथें, जेकर उपयोग बीमार या घायल छात्र मन रायपुर के बड़े अस्पताल (जैसे एम्स रायपुर अउ बालाजी हॉस्पिटल) ले जाए बर तुरंत करे जा सकथें।
> - **आपातकालीन हेल्पलाइन:** आपातकालीन फोन कॉल सुरक्षा मुख्य गेट अउ कैंपस डिस्पेंसरी म चौबीसों घंटा करे जा सकथें, अउ आपातकालीन नंबर मन ला सबो हॉस्टल के lobby अउ सुरक्षा कंट्रोल गेट म चस्पा करे गए हे।
>
> 📄 Source: Page 3 — Medical Center, Dispensary & 24x7 Ambulance

---

### [CG-CONV-09] Chitchat / Out of Scope (Chhattisgarhi) — ✅ SAFE_REFUSAL

- **User Utterance (Spoken):** "जय जोहार! तैं कौन हरस संगी? अउ रायपुर म घुमे बर सबसे बढिया जगा कोन सा हे?"
- **Effective Query (Rewriter):** `What are the popular tourist attractions and places to visit around Naya Raipur and Raipur city?` (Rewritten: True)
- **Top Source Retrieved:** None (Contexts: 0)
- **Latencies:** Retrieval: `3.485s` | LLM: `0.0s` | TTS: `499.9ms` | Total: `4.4s`
- **System Spoken Response:**

> मोला माफ करहू, ये जानकारी अभी के शैक्षणिक दस्तावेज म **नइ मिले हे**।
>
> सुझाव:
>
> - कृपा करके अपन सवाल ल दूसरा शब्द म पूछ के देखहू।
> - सुनिश्चित करहू कि सवाल संस्थान के नियम, प्रवेश, पाठ्यक्रम, कटऑफ या सुविधा ले जुड़े हे।
> - अधिक सहायता बर शैक्षणिक कार्यालय ले संपर्क करहू।

---

### [HI-CONV-01] B.Tech Admissions Process (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "नमस्ते भैया, मुझे यह जानना था कि बीटेक में एडमिशन लेने का पूरा प्रोसेस क्या रहता है?"
- **Effective Query (Rewriter):** `What is the complete admission procedure and eligibility criteria for B.Tech programs at IIIT Naya Raipur?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (B. Tech. Admissions 2026:-  Application for Spot Round) (Contexts: 4)
- **Latencies:** Retrieval: `3.332s` | LLM: `1.871s` | TTS: `861.4ms` | Total: `8.14s`
- **System Spoken Response:**

> नमस्ते! आईआईआईटी नया रायपुर में बीटेक कार्यक्रमों में प्रवेश की प्रक्रिया निम्नलिखित श्रेणियों के माध्यम से होती है:
>
> पहला: ऑल इंडिया कोटा - इसके अंतर्गत पचहत्तर प्रतिशत सीटें ऑल इंडिया के उम्मीदवारों के लिए आरक्षित हैं, और ये सीटें जोसा दो हज़ार पच्चीस और जोसा दो हज़ार छब्बीस काउंसलिंग के माध्यम से भरी जाती हैं।
>
> दूसरा: छत्तीसगढ़ और एनटीपीसी कोटा - छत्तीसगढ़ राज्य से दसवीं और बारहवीं कक्षा पास करने वाले उम्मीदवारों के लिए पचास प्रतिशत सीटें आरक्षित हैं, और एनटीपीसी के साथ कार्यरत कर्मचारियों या उनके पाल्यों के लिए पंद्रह प्रतिशत सीटें आरक्षित हैं। ये सीटें आईआईआईटी नया रायपुर की ऑनलाइन काउंसलिंग प्रक्रिया के माध्यम से भरी जाती हैं। प्रवेश पूरी तरह से जेईई मेन दो हज़ार छब्बीस में उम्मीदवारों के प्रदर्शन से प्राप्त योग्यता के आधार पर होता है।
>
> छत्तीसगढ़ कोटा के तहत आवेदन करने के लिए, उम्मीदवारों को बारहवीं कक्षा की परीक्षा में रसायन विज्ञान, भौतिकी और गणित में कुल मिलाकर पचास प्रतिशत अंक (अनुसूचित जाति, अनुसूचित जनजाति, अन्य पिछड़ा वर्ग और दिव्यांग उम्मीदवारों के लिए पैंतालीस प्रतिशत) प्राप्त होने चाहिए। ऑल इंडिया श्रेणी और एनटीपीसी कोटा के लिए आवेदन करने की पात्रता जोसा दो हज़ार छब्बीस के व्यावसायिक नियमों के अनुसार है।
>
> छत्तीसगढ़ और एनटीपीसी कोटा के तहत प्रवेश चाहने वाले उम्मीदवारों को संस्थान के ऑनलाइन पोर्टल के माध्यम से आवेदन करना होगा। आवेदन प्रक्रिया के तहत उम्मीदवारों को पाँच सौ रुपये का गैर-वापसी योग्य आवेदन शुल्क ऑनलाइन जमा करना होगा। प्राप्त ऑनलाइन आवेदनों और उनके संबंधित जेईई मेन दो हज़ार छब्बीस रैंक के आधार पर प्रत्येक श्रेणी के लिए मेरिट सूची प्रकाशित की जाएगी। पूरी प्रवेश प्रक्रिया ऑनलाइन आयोजित की जाएगी।
>
> 📄 Source: Page 3 — B. Tech. Admissions 2026:-  Application for Spot Round

---

### [HI-CONV-02] Chhattisgarh State Quota (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "क्या छत्तीसगढ़ के मूल निवासियों के लिए कोई रिजर्वेशन या कोटा होता है बीटेक में?"
- **Effective Query (Rewriter):** `What is the reservation policy and domicile quota for Chhattisgarh state residents in B.Tech admissions at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (B. Tech. Admissions 2026:-  Application for Spot Round) (Contexts: 4)
- **Latencies:** Retrieval: `4.011s` | LLM: `2.225s` | TTS: `758.0ms` | Total: `9.33s`
- **System Spoken Response:**

> हाँ, बीटेक कार्यक्रमों के लिए छत्तीसगढ़ के मूल निवासियों के लिए कोटा और रिजर्वेशन का प्रावधान है।
>
> छत्तीसगढ़ कोटा के तहत, छत्तीसगढ़ राज्य के स्कूलों से दसवीं और बारहवीं कक्षा उत्तीर्ण करने वाले उम्मीदवारों के लिए पचहत्तर प्रतिशत सीटें आरक्षित हैं (जो कि कुल सीटों का पचास प्रतिशत बनता है)। इसके अलावा, इस कोटे के अंतर्गत प्रवेश के लिए अनुसूचित जाति, अनुसूचित जनजाति, अन्य पिछड़ा वर्ग और दिव्यांग उम्मीदवारों के लिए छत्तीसगढ़ सरकार के मानदंडों के अनुसार आरक्षण लागू होता है। इसके अतिरिक्त, महिला उम्मीदवारों के लिए छत्तीसगढ़ राज्य सरकार की तीस प्रतिशत महिला आरक्षण की नीति का पालन किया जाता है।
>
> 📄 Source: Page 9 — eligibility will be as per the business rules of JoSAA/CSAB 2026. All

---

### [HI-CONV-03] Hostel Life & Mess Charges (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "हॉस्टल की क्या व्यवस्था है और मेस का खाना और चार्ज कितना पड़ता है हर सेमेस्टर?"
- **Effective Query (Rewriter):** `What are the hostel accommodation facilities, mess food arrangements, and per-semester hostel and mess fee charges at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 6 (M.Tech (DSAI) with CMIT Fellowship 2026) (Contexts: 4)
- **Latencies:** Retrieval: `2.853s` | LLM: `1.526s` | TTS: `554.1ms` | Total: `7.03s`
- **System Spoken Response:**

> नमस्ते! आईआईआईटी नया रायपुर में हॉस्टल और मेस से संबंधित व्यवस्था और शुल्क की जानकारी नीचे दी गई है:
>
> **हॉस्टल और मेस की व्यवस्था:**
>
> - आईआईआईटी नया रायपुर एक सौ प्रतिशत आवासीय परिसर है, जहाँ छात्रों के लिए अलग बहुमंजिला हॉस्टल भवन बने हुए हैं।
> - सभी छात्रों के लिए हॉस्टल में रहना अनिवार्य है।
> - हॉस्टल के कमरों से हाई-स्पीड वाई-फाई और लैन की सुविधा मिलती है।
> - लड़कों और लड़कियों के हॉस्टल के लिए अलग डाइनिंग या मेस की सुविधा है। मेस में शाकाहारी और मांसाहारी दोनों प्रकार का भोजन परोसा जाता है, और मेस का प्रबंधन छात्रों, कर्मचारियों और संकाय सदस्यों की एक समिति द्वारा किया जाता है।
> - इसके अलावा, परिसर में एक केंद्रीय कैंटीन और इनडोर व आउटडोर खेल सुविधाएं तथा जिम जैसी अन्य सुविधाएं भी उपलब्ध हैं।
>
> **मेस और हॉस्टल का शुल्क:** एमटेक (डीएसएआई) कार्यक्रम के संदर्भ में उपलब्ध दस्तावेजों के अनुसार:
>
> - हॉस्टल का शुल्क पंद्रह हज़ार रुपये है।
> - मेस एडवांस पच्चीस हज़ार रुपये है।
>
> 📄 Source: Page 6 — M.Tech (DSAI) with CMIT Fellowship 2026

---

### [HI-CONV-04] Branch Sliding Criteria (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "यार अगर मुझे फर्स्ट ईयर के बाद अपनी ब्रांच बदलकर सीएसई लेनी हो तो कितना सीपीआई लाना पड़ेगा?"
- **Effective Query (Rewriter):** `What is the minimum CPI requirement for branch change to Computer Science and Engineering after the first year at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 6 (Section 5: Branch Sliding & Branch Change Regulations) (Contexts: 1)
- **Latencies:** Retrieval: `3.252s` | LLM: `2.108s` | TTS: `812.3ms` | Total: `8.27s`
- **System Spoken Response:**

> नमस्ते! पहले वर्ष के बाद कंप्यूटर साइंस ऐंड इंजीनियरिंग (सीएसई) में ब्रांच बदलने के लिए आपको दूसरे सेमेस्टर के अंत में कम से कम आठ दशमलव शून्य (आठ दशमलव शून्य) का सीपीआई प्राप्त करना अनिवार्य है। इसके अलावा, आपको अपने पहले और दूसरे सेमेस्टर के सभी पंजीकृत विषयों को पहले प्रयास में बिना किसी सक्रिय बैकलॉग या 'एफ' ग्रेड के उत्तीर्ण करना होगा।
>
> 📄 Source: Page 6 — Section 5: Branch Sliding & Branch Change Regulations

---

### [HI-CONV-05] Medical Attendance Relaxation (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "अगर मेरी तबीयत खराब हो जाए और अटेंडेंस पचहत्तर प्रतिशत से कम हो, तो क्या मेडिकल की कोई छूट मिलती है?"
- **Effective Query (Rewriter):** `What is the attendance policy and medical leave relaxation rule for undergraduate and postgraduate students at IIIT-NR when attendance falls below 75 percent due to illness?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (Section 2: Attendance Regulations & Medical Condonation) (Contexts: 2)
- **Latencies:** Retrieval: `3.691s` | LLM: `1.606s` | TTS: `957.8ms` | Total: `8.31s`
- **System Spoken Response:**

> हाँ, यदि आपकी तबीयत खराब हो जाती है और आपकी अटेंडेंस पचहत्तर प्रतिशत से कम हो जाती है, तो चिकित्सा के आधार पर छूट का प्रावधान है।
>
> सक्षम प्राधिकारी (डीन ऑफ एकेडमिक अफेयर्स) द्वारा सत्यापित चिकित्सा आधारों (गंभीर बीमारी या अस्पताल में भर्ती होने) पर अटेंडेंस में अधिकतम दस प्रतिशत तक की छूट (कंडोनेशन) दी जा सकती है, जिससे न्यूनतम अटेंडेंस पैंसठ प्रतिशत तक हो सकती है।
>
> इस छूट को प्राप्त करने के लिए, छात्र या उनके अभिभावक को कैंपस में लौटने के सात दिनों के भीतर अकादमिक मामलों के कार्यालय में वैध चिकित्सा प्रमाण पत्र और अस्पताल के डिस्चार्ज सारांश के साथ एक औपचारिक चिकित्सा अवकाश आवेदन जमा करना होगा।
>
> 📄 Source: Page 3 — Section 2: Attendance Regulations & Medical Condonation

---

### [HI-CONV-06] Passing Marks & Minimum Grade (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "कॉलेज में पास होने के लिए मिनिमम कौन सा ग्रेड लाना जरूरी होता है?"
- **Effective Query (Rewriter):** `What is the minimum passing grade and SGPA/CGPA required to clear courses at IIIT Naya Raipur?` (Rewritten: True)
- **Top Source Retrieved:** Page 4 (Section 3: 10-Point Letter Grading Scale & Evaluation) (Contexts: 1)
- **Latencies:** Retrieval: `3.32s` | LLM: `1.507s` | TTS: `475.7ms` | Total: `7.29s`
- **System Spoken Response:**

> कॉलेज में किसी भी शैक्षणिक पाठ्यक्रम को पास करने और साफ़ करने के लिए कम से कम डी ग्रेड लाना अनिवार्य होता है, जिसके लिए चार ग्रेड अंक मिलते हैं।
>
> 📄 Source: Page 4 — Section 3: 10-Point Letter Grading Scale & Evaluation

---

### [HI-CONV-07] Hostel Night Curfew (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "रात को हॉस्टल में वापस आने का क्या इन-टाइम या कर्फ्यू रहता है? कितने बजे तक आ सकते हैं?"
- **Effective Query (Rewriter):** `What is the hostel in-time and curfew timing for students returning to the hostel at night at IIIT-NR?` (Rewritten: True)
- **Top Source Retrieved:** Page 4 (Hostel Living & Night Gate Regulations) (Contexts: 2)
- **Latencies:** Retrieval: `3.113s` | LLM: `1.466s` | TTS: `619.5ms` | Total: `7.37s`
- **System Spoken Response:**

> आईआईआईटी नया रायपुर में सभी छात्र-छात्राओं के लिए रात का अनिवार्य कर्फ्यू (इन-टाइम) समय रात्रि दस बजे तक का है। सभी निवासी छात्रों को प्रतिदिन रात्रि दस बजे तक हॉस्टल परिसर में वापस लौटना अनिवार्य है।
>
> 📄 Source: Page 11 — Section 8: Campus Living, Hostel Rules & Healthcare Facilities

---

### [HI-CONV-08] Health Center & Ambulance (Hindi) — ✅ PASS

- **User Utterance (Spoken):** "अगर कैंपस में रात को किसी की अचानक तबीयत बिगड़ जाए तो क्या डॉक्टर और एम्बुलेंस की सुविधा है?"
- **Effective Query (Rewriter):** `Are 24/7 doctor and ambulance emergency medical facilities available on the IIIT-NR campus for students?` (Rewritten: True)
- **Top Source Retrieved:** Page 3 (Medical Center, Dispensary & 24x7 Ambulance) (Contexts: 2)
- **Latencies:** Retrieval: `2.833s` | LLM: `1.636s` | TTS: `868.4ms` | Total: `7.48s`
- **System Spoken Response:**

> हाँ, यदि कैंपस में रात को किसी की अचानक तबीयत बिगड़ जाए तो डॉक्टर और एम्बुलेंस दोनों की सुविधा उपलब्ध है। संस्थान परिसर में एक ऑन-कैंपस मेडिकल डिस्पेंसरी या हेल्थ सेंटर है जहाँ एक निवासी चिकित्सा अधिकारी (डॉक्टर), नर्स और आपातकालीन चिकित्सा परिचारक तैनात रहते हैं। इसके अलावा, छात्रों या कर्मचारियों को रायपुर के प्रमुख अस्पतालों में तुरंत ले जाने के लिए परिसर में चौबीसों घंटे औरतों सप्ताह के सातों दिन एक समर्पित और पूरी तरह से सुसज्जित आपातकालीन एम्बुलेंस तैनात रहती है। आपातकालीन चिकित्सा सहायता के लिए मुख्य सुरक्षा द्वार और कैंपस डिस्पेंसरी पर चौबीसों घंटे संपर्क किया जा सकता है, तथा सभी हॉस्टल लॉबी और सुरक्षा नियंत्रण द्वारों पर आपातकालीन हेल्पलाइन नंबर प्रमुखता से प्रदर्शित किए गए हैं।
>
> 📄 Source: Page 3 — Medical Center, Dispensary & 24x7 Ambulance

---

### [HI-CONV-09] Chitchat / Out of Scope (Hindi) — ⚠️ UNGROUNDED_ANSWER_ON_OOS

- **User Utterance (Spoken):** "अरे सुनो, तुम असली इंसान हो या कोई रोबोट? और यहां कॉलेज का सबसे अच्छा प्रोफेसर कौन है?"
- **Effective Query (Rewriter):** `Who are the faculty members and professors teaching at IIIT Naya Raipur?` (Rewritten: True)
- **Top Source Retrieved:** Page 2 (IIIT Naya Raipur - Comprehensive Faculty Directory) (Contexts: 4)
- **Latencies:** Retrieval: `3.117s` | LLM: `1.39s` | TTS: `657.6ms` | Total: `7.27s`
- **System Spoken Response:**

> नमस्ते! मैं एक आधिकारिक शैक्षणिक सहायक चैटबॉट हूँ, कोई असली इंसान नहीं। आपके दूसरे प्रश्न के संबंध में, उपलब्ध दस्तावेजों में यह जानकारी नहीं दी गई है कि कॉलेज का सबसे अच्छा प्रोफेसर कौन है, क्योंकि दस्तावेजों में केवल संकाय निदेशिका (फैकल्टी डारेक्ट्री) दी गई है जिसमें विभिन्न प्रोफेसरों और उनके विभागों की सूची शामिल है।
>
> 📄 Source: Page 2 — IIIT Naya Raipur - Comprehensive Faculty Directory

---

### [EN-CONV-01] B.Tech Admissions Process (English) — ✅ PASS

- **User Utterance (Spoken):** "Hey there, could you guide me on how to get admission into the B.Tech program at IIIT-NR?"
- **Effective Query (Rewriter):** `Hey there, could you guide me on how to get admission into the B.Tech program at IIIT-NR?` (Rewritten: False)
- **Top Source Retrieved:** Page 3 (B. Tech. Admissions 2026:-  Application for Spot Round) (Contexts: 4)
- **Latencies:** Retrieval: `3.208s` | LLM: `2.005s` | TTS: `138.4ms` | Total: `7.34s`
- **System Spoken Response:**

> Admission to the undergraduate Bachelor of Technology programs at International Institute of Information Technology, Naya Raipur is based strictly on the merit of the candidate as inferred from their performance and rank in the Joint Entrance Examination Main.
>
> Eligible candidates may apply through two main categories:
>
> - First: All-India Quota, where a percentage of seats are filled through the Joint Seat Allocation Authority counselling process.
> - Second: Chhattisgarh and National Thermal Power Corporation Quotas, where fifty percent of seats are reserved for candidates who have passed their tenth and twelfth standard examinations from the State of Chhattisgarh, and fifteen percent of seats are reserved for candidates employed with National Thermal Power Corporation or their wards. These specific quota seats are filled through the institute's online counselling process.
>
> Regarding the eligibility criteria:
>
> - For the Chhattisgarh quota, candidates must secure fifty percent aggregate marks in Physics and Mathematics, and Chemistry depending on the specific program criteria, in their twelfth standard examination, with a relaxed requirement of forty-five percent for Scheduled Caste, Scheduled Tribe, Other Backward Classes, and Persons with Disabilities candidates.
> - For the All-India category and National Thermal Power Corporation quota, the eligibility requirements follow the Joint Seat Allocation Authority guidelines and business rules.
>
> Additionally, there are female-only supernumerary seats and one Kashmiri Migrant seat available in each branch.
>
> 📄 Source: Page 3 — B. Tech. Admissions 2026:-  Application for Spot Round

---

### [EN-CONV-02] Chhattisgarh State Quota (English) — ✅ PASS

- **User Utterance (Spoken):** "I belong to Chhattisgarh state, do we have any reserved state quota seats for B.Tech?"
- **Effective Query (Rewriter):** `I belong to Chhattisgarh state, do we have any reserved state quota seats for B.Tech?` (Rewritten: False)
- **Top Source Retrieved:** Page 9 (eligibility will be as per the business rules of JoSAA/CSAB 2026. All) (Contexts: 4)
- **Latencies:** Retrieval: `2.37s` | LLM: `1.481s` | TTS: `168.8ms` | Total: `5.86s`
- **System Spoken Response:**

> Yes, there are reserved state quota seats for the Bachelor of Technology programs at the institute. Fifty percent of the total seats amount to one hundred thirty-five seats in the Chhattisgarh category, which are divided as forty-five seats in Computer Science and Engineering, forty-five seats in Electronics and Communication Engineering, and forty-five seats in Data Science and Artificial Intelligence.
>
> These seats are reserved for candidates who have passed their tenth and twelfth standard examinations from schools in the State of Chhattisgarh only, and they are filled through the institute's online counselling process. Furthermore, the institute follows the Chhattisgarh State Government policy regarding the reservation of thirty percent of seats for women under this quota.
>
> 📄 Source: Page 3 — B. Tech. Admissions 2026:-  Application for Spot Round

---

### [EN-CONV-03] Hostel Life & Mess Charges (English) — ✅ PASS

- **User Utterance (Spoken):** "Tell me about the hostel life and mess fees, like how much do I actually have to pay per semester?"
- **Effective Query (Rewriter):** `Tell me about the hostel life and mess fees, like how much do I actually have to pay per semester?` (Rewritten: False)
- **Top Source Retrieved:** Page 15 (Fee Type Amount) (Contexts: 4)
- **Latencies:** Retrieval: `2.263s` | LLM: `1.778s` | TTS: `142.8ms` | Total: `6.2s`
- **System Spoken Response:**

> Regarding hostel life, all students are required to stay in the hostel, as the institute is a one-hundred percent residential campus with separate multi-story hostel buildings for male and female students. These hostels are equipped with high-speed Wi-Fi, study rooms, gymnasiums, and dining halls. Additionally, every hostel provides separate sports facilities for indoor games such as table tennis and badminton, and open-air gym facilities with modern equipment are available separately for boys and girls. The mandatory night curfew time for students to return to the campus and enter the hostels is ten o'clock in the evening, and any overnight absences or weekend home visits require an approved night gate pass authenticated by the hostel warden or chief warden.
>
> Concerning the fee structure per semester, the hostel fee is fifteen thousand rupees, and the mess advance is twenty-five thousand rupees. The total fee for the first semester is one lakh eighty-one thousand rupees, which includes a one-time fee with a refundable caution deposit of thirty-one thousand rupees (including a fifteen thousand rupees component), a tuition fee of ninety thousand rupees, an academic and student welfare fee of twenty thousand rupees, alongside the hostel fee and mess advance. For the second semester onwards, the total fee is one lakh fifty thousand rupees.
>
> 📄 Source: Page 15 — Fee Type Amount

---

### [EN-CONV-04] Branch Sliding Criteria (English) — ✅ PASS

- **User Utterance (Spoken):** "Hey bro, what CPI cutoff is required if I want to switch my branch to CSE after the first year?"
- **Effective Query (Rewriter):** `Hey bro, what CPI cutoff is required if I want to switch my branch to CSE after the first year?` (Rewritten: False)
- **Top Source Retrieved:** Page 6 (Section 5: Branch Sliding & Branch Change Regulations) (Contexts: 1)
- **Latencies:** Retrieval: `2.308s` | LLM: `1.131s` | TTS: `93.6ms` | Total: `5.47s`
- **System Spoken Response:**

> To switch your branch to Computer Science and Engineering after the first year, you must have attained a minimum Cumulative Performance Index of eight point zero at the end of the second semester. Additionally, you must have cleared all registered courses in the first and second semesters in the first attempt with no active backlogs or failing grades.
>
> 📄 Source: Page 6 — Section 5: Branch Sliding & Branch Change Regulations

---

### [EN-CONV-05] Medical Attendance Relaxation (English) — ✅ PASS

- **User Utterance (Spoken):** "What happens if someone falls ill and their attendance drops below seventy-five percent? Is there any medical relaxation?"
- **Effective Query (Rewriter):** `What happens if someone falls ill and their attendance drops below seventy-five percent? Is there any medical relaxation?` (Rewritten: False)
- **Top Source Retrieved:** Page 3 (Section 2: Attendance Regulations & Medical Condonation) (Contexts: 1)
- **Latencies:** Retrieval: `2.28s` | LLM: `1.287s` | TTS: `169.2ms` | Total: `5.77s`
- **System Spoken Response:**

> Yes, there is medical relaxation available. A relaxation or condonation of up to a maximum of ten percent in attendance, allowing a minimum attendance floor of sixty-five percent, may be granted by the Competent Authority, which is the Dean of Academic Affairs, on verified medical grounds such as serious illness or hospitalization.
>
> To avail of this, the student or their guardian must submit a formal medical leave application supported by valid medical certificates and hospital discharge summaries to the Academic Affairs Office within seven days of returning to campus.
>
> However, if a student's attendance falls below seventy-five percent, or below sixty-five percent even after approved medical condonation, they will be summarily debarred from appearing in the End-Semester Examination of that course and shall be awarded a failing grade due to attendance shortage. Such a course must then be repeated in a subsequent semester.
>
> 📄 Source: Page 3 — Section 2: Attendance Regulations & Medical Condonation

---

### [EN-CONV-06] Passing Marks & Minimum Grade (English) — ✅ PASS

- **User Utterance (Spoken):** "What is the minimum passing letter grade required to clear a course here?"
- **Effective Query (Rewriter):** `What is the minimum passing letter grade required to clear a course here?` (Rewritten: False)
- **Top Source Retrieved:** Page 4 (Section 3: 10-Point Letter Grading Scale & Evaluation) (Contexts: 1)
- **Latencies:** Retrieval: `2.865s` | LLM: `1.137s` | TTS: `42.0ms` | Total: `6.06s`
- **System Spoken Response:**

> The minimum letter grade required to pass and clear any academic course is grade 'D', which corresponds to four grade points.
>
> 📄 Source: Page 4 — Section 3: 10-Point Letter Grading Scale & Evaluation

---

### [EN-CONV-07] Hostel Night Curfew (English) — ✅ PASS

- **User Utterance (Spoken):** "Is there a curfew or in-time for the hostels at night? When do the gates close?"
- **Effective Query (Rewriter):** `Is there a curfew or in-time for the hostels at night? When do the gates close?` (Rewritten: False)
- **Top Source Retrieved:** Page 4 (Hostel Living & Night Gate Regulations) (Contexts: 2)
- **Latencies:** Retrieval: `1.951s` | LLM: `1.632s` | TTS: `56.7ms` | Total: `5.65s`
- **System Spoken Response:**

> Yes, there is a mandatory curfew and in-time for the hostels at night. All resident students must return to the hostel campus by ten o'clock in the evening daily.
>
> 📄 Source: Page 11 — Section 8: Campus Living, Hostel Rules & Healthcare Facilities

---

### [EN-CONV-08] Health Center & Ambulance (English) — ✅ PASS

- **User Utterance (Spoken):** "What medical support is available on campus in case of a late-night health emergency?"
- **Effective Query (Rewriter):** `What medical support is available on campus in case of a late-night health emergency?` (Rewritten: False)
- **Top Source Retrieved:** Page 3 (Medical Center, Dispensary & 24x7 Ambulance) (Contexts: 2)
- **Latencies:** Retrieval: `2.144s` | LLM: `1.678s` | TTS: `181.0ms` | Total: `5.96s`
- **System Spoken Response:**

> In the event of a late-night health emergency on campus, the following medical support and facilities are available:
>
> - An on-campus medical dispensary and health center is available, which is staffed by a resident medical officer, a nurse, and an emergency medical attendant to provide first aid, routine consultations, and essential medications.
> - A dedicated and fully equipped emergency ambulance is stationed on campus twenty-four hours a day, seven days a week, to provide rapid emergency transport for students or staff to premier hospitals in Raipur, such as All India Institute of Medical Sciences Raipur and Balaji Hospital.
> - Emergency medical helpline numbers can be called around the clock directly at the security main gate and the campus dispensary, and these numbers are also prominently posted in all hostel lobbies and security control gates.
>
> 📄 Source: Page 3 — Medical Center, Dispensary & 24x7 Ambulance

---

### [EN-CONV-09] Chitchat / Out of Scope (English) — ✅ SAFE_REFUSAL

- **User Utterance (Spoken):** "Hey! Are you an AI or a real person? Also can you tell me a good place to hangout in Raipur?"
- **Effective Query (Rewriter):** `Hey! Are you an AI or a real person? Also can you tell me a good place to hangout in Raipur?` (Rewritten: False)
- **Top Source Retrieved:** None (Contexts: 0)
- **Latencies:** Retrieval: `2.341s` | LLM: `0.0s` | TTS: `82.2ms` | Total: `4.35s`
- **System Spoken Response:**

> I apologize, this information was **not found** in the currently available academic documents.
>
> Suggestions:
>
> - Try rephrasing your question with different academic keywords.
> - Ensure your inquiry pertains to institute regulations, admissions, curriculum, cutoffs, or campus facilities.
> - Contact the academic affairs office for further assistance.

---
