

## **📌 Ms-AeDNet**

A Multi-scale Attention-enhanced Dynamic Network for Multi-step Performeace Prediction of Hydrogen Proton Exchange Membrane Fuel Cells

## **🔍 Abstract**

The global transition to sustainable, low-carbon energy systems positions hydrogen energy as a pivotal solution for mitigating climate change and enhancing energy security. Among the most promising technologies for clean hydrogen utilization are Proton Exchange Membrane Fuel Cells (PEMFCs), known for their high efficiency and zero emissions. However, the long-term durability and performance prediction of PEMFCs remain major challenges, hindering their widespread adoption in critical sectors. Traditional methods struggle to capture the multi-scale temporal dependencies and the dynamic effects of external operating conditions. To address these challenges, we propose a Multi-scale Attention-enhanced Dynamic Network (Ms-AeDNet) that integrates global adaptive decomposition, local time enhancement, and cross-factor attentive fusion. This solution enhances the performance prediction accuracy of PEMFCs by leveraging deep learning to model degradation patterns and environmental influences simultaneously. Validated on two industrial datasets, Ms-AeDNet outperforms baseline models, reducing MAE and MSE by 27.63\% and 37.48\% on average. Functional contribution analysis and visualization studies further confirm its robustness across varying operating conditions, making it a reliable tool for predictive maintenance and life cycle optimization of hydrogen PEMFCs.

## **⚙️ Start**

```bash

git clone https://github.com/BMRETURN/Ms-AeDNet.git
cd Ms-AeDNet

pip install -r requirements.txt

python main.py --model_name MsAeDNet
```