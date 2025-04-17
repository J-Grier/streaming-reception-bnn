# 🎬 Predicting TV Episode Reception with Bayesian Neural Networks

## Overview
This project applies Bayesian Neural Networks (BNNs) combined with topic modeling and sentiment analysis to predict how audiences will receive streaming TV series. By analyzing fan discussions on Reddit, the model forecasts episode-level audience ratings with probabilistic confidence, providing actionable insights for content creators and marketers.

## Objectives
- **Primary Goal**: Predict episode-level audience reception with quantified uncertainty.
- **Actionable Insights**: Enable strategic content and marketing decisions based on predicted audience reactions.

## Approach
The analysis employs a multi-layered Bayesian approach:
- **Bayesian Neural Network (BNN)**: Uses Pyro to predict IMDb episode ratings, modeling uncertainty explicitly.
- **Latent Dirichlet Allocation (LDA)**: Extracts topics from Reddit discussions, giving context to viewer sentiment.
- **VADER Sentiment Analysis**: Assesses emotional tone, capturing sentiment nuances from audience comments.

The combination of these tools provides not only predictions about audience reception but also detailed insights into why certain reactions occur.

## Key Findings
- **Reliable Predictions with Uncertainty**: BNN effectively predicted episode ratings for diverse shows like *Severance*, *The Bear*, and *Ahsoka*, accurately reflecting audience sentiment.
- **Detection of Cultural Outliers**: Clearly identified shows like *Velma*, where predicted scores significantly differed from actual IMDb ratings due to unique audience dynamics like review bombing or meme-driven backlash.
- **Effective Generalization**: Successfully predicted reception trends within shows, even for episodes not explicitly shown to the model during training.

## Impact and Applications
- **Strategic Content Development**: Allows producers to proactively adjust content and marketing strategies based on predictive insights.
- **Enhanced Audience Targeting**: Facilitates targeted promotional campaigns by identifying key sentiment drivers.
- **Risk Mitigation**: Early identification of potentially controversial or poorly received content.

## Limitations and Future Directions
- **Platform-Specific Data**: The current model primarily analyzes Reddit data, potentially limiting broader applicability.
- **Future Enhancements**: Incorporate additional platforms (e.g., Twitter, YouTube) and explore further hybrid modeling techniques.

---

## 📚 Data Sources

| Feature Set         | Description                            |
|---------------------|----------------------------------------|
| Reddit comments     | Episode-specific threads from r/TV etc |
| LDA topics          | 10-dimensional vectors per episode     |
| VADER sentiment     | Average tone per episode               |
| IMDb labels         | Soft labels (IMDb scores scaled 0–1)   |

Reddit was used to reflect real-time discourse. Sentiment and topic modeling were computed per episode to form the full feature set.

---

## 🧠 Bayesian Neural Networks: A Brief Mathematical Primer

A **Bayesian Neural Network** places **distributions over weights** rather than point estimates. Instead of learning a fixed weight vector `w`, we learn a **posterior distribution**:

`p(w | D) ∝ p(D | w) ⋅ p(w)`

Where:
- `p(w)` is the prior (e.g., standard Normal)
- `p(D | w)` is the likelihood (e.g., Normal for regression)
- `p(w | D)` is the posterior

We use **variational inference** with Pyro to approximate this posterior. The loss function is the **Evidence Lower Bound (ELBO)**:

`ELBO = E_q(w)[log p(D | w)] - KL(q(w) || p(w))`

We used `AutoMultivariateNormal` to capture full posterior correlation.

---

## 🛠️ Model Pipeline

1. **Preprocessing**
   - Cleaned and tokenized Reddit text
   - Generated LDA topic vectors (10D)
   - Computed average VADER sentiment
2. **Model Inputs**
   - 11D input vector: `[topic_0, ..., topic_9, vader_sentiment]`
3. **BNN Training**
   - Bayesian regression with Normal likelihood
   - 10,000-step ELBO optimization via Pyro
4. **Posterior Inference**
   - `Predictive(model, guide, num_samples=1000)`
   - Posterior mean and std dev for each episode prediction

---

## 📊 Results Overview

The model performed remarkably well, predicting IMDb reception across episodes and genres with high precision and calibrated uncertainty.

### 🎯 Episode-Level Accuracy

- Most episodes were predicted **within ±1 std dev** of their true score
- Shows like *The Bear*, *Severance*, and *White Lotus* had very tight fits
- **Velma** was consistently misaligned — predicted around 0.72 while true scores were 0.27–0.32, revealing discourse/ratings mismatch

### 🧪 Generalization Test

We trained the model on all but the **last 2 episodes per show**, then evaluated generalization:

- ✅ *One Piece* and *Fallout* episodes were successfully predicted, even with sparse Reddit data
- ✅ *Ahsoka*, *The Boys*, *Daredevil*, and *Bluey* all showed strong performance
- ❌ *Velma* remained outside model bounds

---

## 📈 Visualization

We plotted predictions with uncertainty and true ratings:

![Predicted vs Actual](Assets/Plots/bnn_predicted_vs_actual_all.png)

- Vertical bars = BNN uncertainty  
- Blue dots = predicted scores  
- Xs = IMDb labels

---

## ✅ Conclusion

This project demonstrates how Bayesian modeling can surface nuance in audience reception that traditional classifiers might miss.

The Bayesian Neural Network quantified uncertainty, identified outlier behavior (Velma), and generalized across shows with sparse discourse (One Piece and Fallout). This project provided hands-on experience in:

- Probabilistic programming with Pyro  
- Feature engineering from unstructured text  
- Calibrated evaluation using soft labels and error bounds

This approach could scale to monitor media reception in real-time, flag controversial content, or help streaming platforms align discourse with audience ratings.

---

## 🔄 Future Work

| Idea                             | Goal                                    |
|----------------------------------|-----------------------------------------|
| Add YouTube comments             | Better signal for polarizing shows      |
| Incorporate time-series info     | Model reception shifts over episode drops |
| Compare against MAP (non-Bayesian) | Evaluate uncertainty vs hard prediction |
| Use LDA+LSTM hybrid              | Learn long-term thematic arcs           |

---

## 🧾 References

- [IMDb](https://www.imdb.com/)
- [Pyro](https://pyro.ai/)
- [Reddit](https://reddit.com)
