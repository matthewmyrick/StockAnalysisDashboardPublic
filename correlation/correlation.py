import numpy as np
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy import cov, mean, std
from numpy.linalg import eig
from scipy.stats import pearsonr, spearmanr
from sklearn import preprocessing
from api.fmp_request import *


def correlation_analysis(symbols):
    stockList = symbols.split(",")
    stockList.reverse()
    response = get_bars(symbols=symbols)
    stock1Df = pd.DataFrame(json.loads(response)["historicalStockList"][0]['historical'])
    stock1Df = stock1Df[::-1]
    stock1 = np.array(
        stock1Df['changePercent'].to_list()
    )
    stock2Df = pd.DataFrame(json.loads(response)["historicalStockList"][1]['historical'])
    stock2Df = stock2Df[::-1]
    stock2 = np.array(
        stock2Df['changePercent'].to_list()
    )

    if (len(stock1) > len(stock2)):
        stock1 = stock1[(len(stock1) - len(stock2)):]
    elif (len(stock1) < len(stock2)):
        stock2 = stock2[(len(stock2) - len(stock1)):]

    covariance = cov(stock1, stock2)
    ew,ev = eig(covariance)
    correlationPearson, p_valPearson = pearsonr(stock1, stock2)
    correlationSpearman, p_valSpearman = spearmanr(stock1, stock2)
    return stock1, stock2, covariance, ew, ev, correlationPearson, p_valPearson, correlationSpearman, p_valSpearman

def correlation_hypothesis(confidence_int, corrPearson, corrSpearman):
    alpha = 0.05
    pearsonCorrAnalysis = ""
    pearsonConfAnalysis = ""

    if (confidence_int[0] < alpha):
        pearsonConfAnalysis = "Since the p  value ({}) < alpha ({}) we can conclude the correlation is statistically" \
                              " significant.".format(round(confidence_int[0], 4), alpha)
    else:
        pearsonConfAnalysis = "Since the p value ({}) > alpha ({}) we can conclude the correlation is NOT statistically" \
                              " significant ".format(round(confidence_int[0], 4), alpha)

    if (-1 <= corrPearson <= -0.7):
        pearsonCorrAnalysis = "Strong Negative Correlation."
    elif (-0.7 < corrPearson <= -0.3):
        pearsonCorrAnalysis = "Moderate Negative Correlation."
    elif (-0.3 < corrPearson < 0.3):
        pearsonCorrAnalysis = "No Correlation."
    elif (0.3 <= corrPearson < 0.7):
        pearsonCorrAnalysis = "Moderate Positive Correlation."
    elif (0.7 <= corrPearson <= 1):
        pearsonCorrAnalysis = "Strong Positive Correlation."
    else:
        pearsonCorrAnalysis = "Error Conducting Analysis."


    spearmanCorrAnalysis = ""
    spearmanConfAnalysis = ""

    if (confidence_int[1] < alpha):
        spearmanConfAnalysis = "Since the p value ({}) < alpha ({}) we can conclude the correlation is statistically" \
                              " significant.".format(round(confidence_int[1], 4), alpha)
    else:
        spearmanConfAnalysis = "Since the p value ({}) > alpha ({}) we can conclude the correlation is NOT statistically" \
                              " significant ".format(round(confidence_int[1], 4), alpha)

    if (-1 <= corrSpearman <= -0.7):
        spearmanCorrAnalysis = "Strong Negative Correlation."
    elif (-0.7 < corrSpearman <= -0.3):
        spearmanCorrAnalysis = "Moderate Negative Correlation."
    elif (-0.3 < corrSpearman < 0.3):
        spearmanCorrAnalysis = "No Correlation."
    elif (0.3 <= corrSpearman < 0.7):
        spearmanCorrAnalysis = "Moderate Positive Correlation."
    elif (0.7 <= corrSpearman <= 1):
        spearmanCorrAnalysis = "Strong Positive Correlation."
    else:
        spearmanCorrAnalysis = "Error Conducting Analysis."


    return pearsonCorrAnalysis, pearsonConfAnalysis, spearmanCorrAnalysis, spearmanConfAnalysis