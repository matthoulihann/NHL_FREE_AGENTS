# NHL_FREE_AGENTS
🏒 NHL Free Agent Evaluation Models
This repository contains machine learning models and data pipelines used to evaluate NHL free agents by predicting contract term, cap hit percentage, and on-ice value (GAR). The end goal is to help identify under- or over-valued players based on their projected performance and fair market value.

🔧 Project Components
1. Contract Term Prediction
Model: Random Forest Regressor

Target: term (contract length in years)

Features:

Age as of July 1, 2024

UFA/RFA status

Years of NHL experience

Historical WAR, GAR, and TOI

Training Data: NHL contracts signed from 2018–2024

Use Case: Predict expected contract length for 2024 UFAs/RFAs based on market precedent

2. Cap Hit % Prediction
Model: Random Forest Regressor

Target: prev_cap_hit_pct (cap hit / salary cap in signing year)

Features:

Weighted WAR, GAR, and TOI (3-year span)

Age, UFA/RFA status, and NHL experience

Output: Predicted cap hit % → converted to AAV using a $95.5M salary cap

3. GAR Projection (2024–25 Season)
Model: Ridge Regression

Target: projected_gar_24_25

Features:

Historical GAR, WAR (from 2021–22 to 2023–24)

Weighted by recency

TOI per game and games played factored in

4. Fair Market Value System
Method:

Calculate average $/GAR across historical free agent signings

Multiply projected GAR × current $/GAR to get estimated fair value

Value Score = percentile ranking scaled from 0 to 100

📊 Data Sources
Source	Description
CapFriendly	Contract data (AAV, term, UFA/RFA status)
NHL.com	Player profiles, career totals, debut years
Evolving Hockey	WAR, GAR, SPAR metrics
Spotrac	Supplemental contract details (2018–2024)
Manually cleaned CSVs	Free agent lists, projection tables, experience data

💻 Tech Stack
Languages: Python, SQL

Libraries: pandas, scikit-learn, matplotlib, seaborn

Database: MySQL (via Railway)

Frontend: Next.js + Tailwind CSS

Deployment: Vercel

