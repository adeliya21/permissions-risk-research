# Third Party Apps Graph API Permissions Risk

Third-party applications integrated with enterprise platforms such as Microsoft Graph and Office 365 introduce significant security risks due to over-privileged access and lack of visibility. 

This code presents an unsupervised machine learning framework for detecting anomalous and potentially risky third-party applications based on their permissions. Using an anonymized dataset of application permissions, we engineer security-relevant features and apply three anomaly detection techniques: 
- Isolation Forest,
- One-Class Support Vector Machine (SVM),
- and DBSCAN clustering.

Results demonstrate that combining multiple models improves detection robustness and highlights high-risk applications exhibiting excessive or sensitive permission usage. The proposed approach provides a scalable and automated mechanism for security teams to prioritize application reviews and reduce attack surfaces.
