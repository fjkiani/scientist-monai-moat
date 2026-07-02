# Informed Consent Template — Oncology-Arbiter Investigational Study

**Institution**: `[[STUDY_INSTITUTION]]`
**IRB Number**: `[[IRB_NUMBER]]`
**Principal Investigator**: `[[PI_NAME]], [[PI_DEGREES]]`
**24-hour PI contact**: `[[PI_PHONE]]`
**Version**: `[[CONSENT_VERSION]]`, `[[CONSENT_DATE]]`

---

## Consent to participate in the "Oncology-Arbiter Reader-Augmentation Study"

### 1. Why am I being asked to consider this study?

You are being asked to consider participation because you had a screening mammogram at `[[STUDY_SITE]]` between `[[START_DATE]]` and `[[END_DATE]]` and meet the study's eligibility criteria.

### 2. What is the purpose of the study?

The purpose of this research is to evaluate whether an investigational artificial-intelligence software system called **oncology-arbiter** can help radiologists reduce unnecessary follow-up imaging and biopsies without missing cancer.

### 3. What is oncology-arbiter?

You are being asked to consent to your mammogram being analyzed by an investigational AI system (oncology-arbiter). This system is **RESEARCH USE ONLY** and has **not** been approved or cleared by the U.S. Food and Drug Administration for use in medical decision-making. It is investigational software developed for the sole purpose of scientific research. During this study, the AI system's predictions will **NOT** be shared with your radiologist during your routine clinical care. **Your medical care will not depend on this system, and no decisions about your health will be made using its outputs.**

The AI system was trained on breast imaging and pathology datasets from other institutions, not from `[[STUDY_INSTITUTION]]`. Its performance on your images has not been validated. Even after this study, its use in routine clinical care is not authorized.

### 4. What will happen if I take part?

If you consent:
- Your existing mammogram images and the radiologist's report will be included in a research dataset.
- The oncology-arbiter system will process your images and generate a prediction (a risk score for the presence of a lesion). This prediction will be stored in a research database at `[[STUDY_INSTITUTION]]` and **will not be shared with your treating clinicians**.
- Approximately 12 months after your mammogram, we will look at your medical record to see whether you were called back for additional imaging, had a biopsy, and if so, what the biopsy showed. We will compare that outcome to what the AI system predicted.
- Your name, medical record number, date of birth, and address will **not** leave `[[STUDY_INSTITUTION]]`. We will assign a study identifier (a random number) that cannot be used to identify you outside of the study.

**There will be no additional visits, no additional imaging, and no additional procedures.** Your care will follow your radiologist's normal recommendations.

### 5. Are there any risks?

The primary risk is the small possibility of accidental disclosure of your health information (Section 8 below). We take strong steps to prevent this. There is no physical risk because we are not doing any additional imaging or procedures.

Because the AI system's predictions are **not** shared with your treating clinicians during this study, there is no risk that the AI's output — correct or incorrect — will affect your care.

### 6. Are there any benefits?

There is **no direct medical benefit to you** from participating. The findings of this study may help future patients by identifying whether AI systems like oncology-arbiter can safely reduce unnecessary follow-up.

### 7. What are the alternatives?

You may choose not to participate. Your decision will not affect the care you receive at `[[STUDY_INSTITUTION]]` in any way.

### 8. How will my information be protected? (HIPAA §164.508 Authorization)

Under the Health Insurance Portability and Accountability Act (HIPAA), we must ask your permission (authorization) to use and disclose your health information for this research. This section covers the seven required elements of a HIPAA authorization:

- **8.1 Information to be used or disclosed**: your screening mammogram images (DICOM files), the radiologist's interpretation and BI-RADS assessment, and any follow-up imaging or biopsy results from the 12 months following your mammogram.
- **8.2 Purpose of the use or disclosure**: to determine whether the oncology-arbiter software can help radiologists interpret screening mammograms more accurately. This is **research** purpose only.
- **8.3 Recipient of the disclosure**: the research team at `[[STUDY_INSTITUTION]]` under the direction of `[[PI_NAME]]`. De-identified data (with study identifier only) may be shared with collaborating institutions listed in `[[COLLABORATOR_LIST]]` under a Data Use Agreement.
- **8.4 Expiration**: this authorization will expire on `[[AUTHORIZATION_EXPIRATION_DATE]]` or at the end of the study, whichever is later.
- **8.5 Right to revoke**: you may **revoke this authorization at any time by writing to `[[HRPP_ADDRESS]]`**. If you revoke, we will stop using your information as of the date of your revocation, but we may not be able to remove information that has already been used in the analysis. Revocation does not affect any care you have received or will receive at `[[STUDY_INSTITUTION]]`.
- **8.6 Treatment not conditioned on authorization**: your treatment, payment, enrollment in a health plan, or eligibility for benefits will **not** be conditioned on whether you sign this authorization.
- **8.7 Potential for re-disclosure**: information disclosed to the research team is covered by federal privacy laws, but once it is disclosed under this authorization it may be subject to re-disclosure by the recipient and may no longer be protected by federal privacy laws. The research team takes technical safeguards described in Section 9 to minimize this risk.

### 9. Data security

Your information will be stored in a HIPAA-compliant enclave at `[[STUDY_INSTITUTION]]` with access limited to the study team. Access is controlled by role-based permissions (§164.312(a)), audit-logged (§164.312(b)), and transmitted only over encrypted channels (§164.312(e)). We follow the site's incident response policy `[[SITE_INCIDENT_RESPONSE_POLICY]]` for any suspected breach.

### 10. Who to contact

- **About the research**: `[[PI_NAME]]` at `[[PI_PHONE]]` or `[[PI_EMAIL]]`.
- **About your rights as a research participant**: `[[IRB_NAME]]` at `[[IRB_PHONE]]` or `[[IRB_EMAIL]]`.

### 11. Consent

I have read this consent form (or had it read to me). I have had a chance to ask questions and my questions have been answered. I understand that participation is voluntary, that I may withdraw at any time, and that my decision will not affect my care.

I consent to participate in this research study and I authorize the use and disclosure of my health information as described above.

---

Participant name (print): _______________________________________________

Participant signature: __________________________________  Date: __________

Person obtaining consent (print): _______________________________________

Person obtaining consent signature: _____________________  Date: __________

---

*This template must be reviewed by `[[IRB_NAME]]` before use. Bracketed placeholders must be filled by the sponsoring institution. The language in Section 3 (`RESEARCH USE ONLY`, "not been approved by the FDA", "your care will not depend on this system") derives from `src/oncology_arbiter/__init__.py::RUO_DISCLAIMER` and must not be softened.*
