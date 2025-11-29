# Cyber Threat Intelligence Datasets with MITRE ATT&CK Labels

## Summary
This document lists publicly available datasets suitable for training BERT models for TTP (Tactics, Techniques, Procedures) extraction and classification using the MITRE ATT&CK framework.

---

## 1. **sarahwei/cyber_MITRE_attack_tactics-and-techniques**

**Platform:** Hugging Face  
**Direct URL:** https://huggingface.co/datasets/sarahwei/cyber_MITRE_attack_tactics-and-techniques

**Size:** 654 samples

**Content:** Question-answering dataset about MITRE ATT&CK tactics and techniques
- Contains descriptions of tactics (14 tactics total)
- Contains descriptions of techniques with IDs (e.g., T1071, T1059, T1548, etc.)
- Each entry includes technique ID, name, description, tactics, affected platforms, and detection methods

**Label Format:** 
- Technique IDs (T-codes like T1548, T1134, T1087, etc.)
- Tactic names (Collection, Command and Control, Credential Access, etc.)
- Includes sub-techniques (e.g., T1548.006, T1134.002)

**Quality/Usability:**
- ✅ Well-structured Q&A format
- ✅ Based on official MITRE ATT&CK v15
- ❌ Small size (only 654 samples)
- ✅ Good for understanding MITRE taxonomy
- ⚠️ Limited to Q&A format, not actual threat reports

**Use Case:** Training/fine-tuning for MITRE ATT&CK knowledge, but too small for main training

---

## 2. **Zainabsa99/mitre_attack**

**Platform:** Hugging Face  
**Direct URL:** https://huggingface.co/datasets/Zainabsa99/mitre_attack

**Size:** 508 samples

**Content:** MITRE ATT&CK technique descriptions
- Technique ID (T-codes)
- Technique name and full description
- Creation/modification dates
- Attack type (enterprise-attack)
- Tactics
- Detection methods

**Label Format:**
- Technique IDs with sub-techniques (e.g., T1578.005, T1071.003, T1055.003)
- Single or multiple tactic labels per technique
- Includes detailed detection and mitigation information

**Quality/Usability:**
- ✅ Clean structured format
- ✅ Comprehensive technique descriptions
- ❌ Very small (only 508 samples)
- ✅ Good quality descriptions from official MITRE
- ⚠️ Not actual CTI reports, just MITRE documentation

**Use Case:** Reference dataset for MITRE ATT&CK techniques, supplementary training data

---

## 3. **mrmoor/cyber-threat-intelligence**

**Platform:** Hugging Face  
**Direct URL:** https://huggingface.co/datasets/mrmoor/cyber-threat-intelligence

**Size:** 9,732 samples

**Content:** Real cyber threat intelligence reports and articles
- Threat reports from security vendors
- Malware analysis
- Attack campaign descriptions
- APT group activities

**Label Format:**
- Entity annotations (malware, threat-actor, identity, location, TIME, attack-pattern)
- Some samples include MITRE technique references
- Multi-entity labeled spans with start/end offsets

**Quality/Usability:**
- ✅ Large dataset with actual CTI text
- ✅ Real-world threat intelligence reports
- ✅ Named entity recognition (NER) labels
- ⚠️ Not all samples have explicit MITRE ATT&CK labels
- ✅ Suitable for pre-training on CTI domain
- ✅ Contains context about real attacks

**Use Case:** Excellent for CTI domain adaptation, but needs MITRE label mapping

---

## 4. **Additional Hugging Face Datasets** (Found but less suitable)

### Fmfawaz32/mitre-attack & Fmfawaz32/mitre-attack-dataset
- Small datasets
- Limited information available
- Need manual inspection

### Tejeswara/cybersec_mitre_attack_tactics_techniques_instruct
- Instructional dataset
- ~3.2k samples
- May be suitable for instruction-tuning

### khangmacon/mitreattackQA
- Question-answering format
- Smaller dataset
- Similar to sarahwei dataset

---

## 5. **Kaggle Datasets**

### Cybersecurity Attack Dataset (tannubarot)
**URL:** https://www.kaggle.com/datasets/tannubarot/cybersecurity-attack-and-defence-dataset

**Content:** Cybersecurity attack and defense data
- May contain attack scenarios
- Need to verify MITRE ATT&CK labels

### CVE and CWE mapping Dataset (krooz0)
**URL:** https://www.kaggle.com/datasets/krooz0/cve-and-cwe-mapping-dataset

**Content:** CVE (Common Vulnerabilities and Exposures) mapped to CWE
- 2021 dataset
- May contain some MITRE mappings
- More focused on vulnerabilities than TTPs

### CREMEv2 Datasets
**URL:** https://www.kaggle.com/datasets/masjohncook/cremev2-datasets

**Content:** Cybersecurity-related dataset
- Need manual verification for MITRE content

---

## 6. **GitHub Repositories** (Potential Dataset Sources)

Based on search results, there are repositories related to MITRE ATT&CK, but specific dataset links need manual exploration:

- MITRE ATT&CK integration projects
- Elastic SIEM integration datasets
- Security analytics datasets

**Recommended Actions:**
- Search GitHub for: "MITRE ATT&CK dataset"
- Look for CTI sharing projects
- Check MITRE's official repository

---

## 7. **Academic & Research Datasets**

### TRAM (Threat Report ATT&CK Mapping)
**Organization:** MITRE
**URL:** https://github.com/center-for-threat-informed-defense/tram

**Description:** Tool for mapping threat reports to ATT&CK
- May include training data
- Check for publicly available annotated datasets

### UCF TRAM Dataset
**Research:** Various academic papers on TTP extraction
- Check papers on "TTP extraction from CTI reports"
- Look for publicly shared annotated corpora

---

## Dataset Recommendations for Your Use Case

### Best Options:

1. **Primary Training Data:**
   - **mrmoor/cyber-threat-intelligence** (9,732 samples)
     - Largest dataset with real CTI text
     - Good for domain adaptation
     - Needs MITRE label enrichment

2. **MITRE Knowledge Base:**
   - **sarahwei/cyber_MITRE_attack_tactics-and-techniques** (654 samples)
   - **Zainabsa99/mitre_attack** (508 samples)
     - Use together for ~1,200 MITRE technique descriptions
     - Good for fine-tuning on MITRE taxonomy

3. **Supplementary:**
   - Kaggle CTI datasets (need verification)
   - Custom scraping from public CTI reports

### Recommended Approach:

1. **Phase 1 - Domain Adaptation:**
   - Use mrmoor/cyber-threat-intelligence for initial BERT fine-tuning
   - This teaches the model CTI language and entity recognition

2. **Phase 2 - MITRE Mapping:**
   - Combine sarahwei + Zainabsa99 datasets
   - Create synthetic training data by mapping entities to MITRE techniques
   - Use data augmentation on MITRE descriptions

3. **Phase 3 - Augmentation:**
   - Scrape public CTI blogs (Talos, Unit 42, Kaspersky)
   - Use ChatGPT/GPT-4 to generate synthetic TTP-labeled text
   - Manual annotation of high-value samples

---

## Data Augmentation Strategies

### To Overcome Small Dataset Size:

1. **Back-translation:** Translate to other languages and back to English
2. **Paraphrasing:** Use GPT-4 to rephrase technique descriptions
3. **Template-based generation:** Create attack scenario templates
4. **Synonym replacement:** For technical terms
5. **Contextual word embeddings:** Generate similar sentences

---

## Additional Resources to Explore

### CTI Report Sources (for manual annotation):
- **Unit 42 Threat Research:** https://unit42.paloaltonetworks.com/
- **Talos Intelligence:** https://blog.talosintelligence.com/
- **Kaspersky Securelist:** https://securelist.com/
- **Mandiant:** https://www.mandiant.com/resources/blog
- **CrowdStrike:** https://www.crowdstrike.com/blog/

### Tools for Dataset Creation:
- **TRAM:** Automated TTP mapping tool
- **ATT&CK Navigator:** Visualization and mapping
- **Jupyter notebooks:** Many Kaggle notebooks for CTI analysis

---

## Limitations & Challenges

### Current Dataset Limitations:
- ❌ No single large dataset with >5,000 CTI reports + MITRE labels
- ❌ Most datasets are either:
  - Large but unlabeled CTI text (mrmoor)
  - Small but well-labeled MITRE descriptions (sarahwei, Zainabsa99)
- ⚠️ Multi-label classification is challenging with small data
- ⚠️ Class imbalance (some techniques rarely mentioned)

### Solutions:
1. **Combine multiple datasets**
2. **Use transfer learning** from general NER models
3. **Weak supervision** with rule-based initial labels
4. **Active learning** to prioritize manual annotation
5. **Semi-supervised learning** with pseudo-labels

---

## Conclusion

While there's **no perfect large-scale dataset** with 5,000+ pre-labeled CTI reports, you can create a working solution by:

1. **Starting with:** mrmoor/cyber-threat-intelligence (9.7k samples)
2. **Adding MITRE knowledge:** sarahwei + Zainabsa99 datasets (~1.2k)
3. **Augmenting with:** Synthetic data + manual annotation
4. **Target:** Aim for 3,000-5,000 quality samples after augmentation

**Expected Performance:**
- With 3,000+ mixed samples: 60-75% F1 score
- With 5,000+ samples + augmentation: 70-85% F1 score
- State-of-the-art systems: 75-90% F1 (with larger proprietary datasets)

**Next Steps:**
1. Download and explore the top 3 datasets
2. Analyze label quality and coverage
3. Design data augmentation pipeline
4. Consider semi-supervised or few-shot learning approaches

