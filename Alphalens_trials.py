# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 00:17:49 2019

@author: Sudhin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:21:54 2019

@author: Sudhin
"""
#https://github.com/quantopian/alphalens/blob/master/alphalens/performance.py
#https://www.quantopian.com/tutorials/alphalens#lesson2
#https://blog.quantinsti.com/introduction-zipline-python/
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#https://scikit-learn.org/stable/modules/ensemble.html
#https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
#https://builtin.com/artificial-intelligence/ai-trading-stock-market-tech
#https://medium.com/datadriveninvestor/the-future-of-trading-belong-to-artificial-intelligence-a4d5887cb677
#https://towardsdatascience.com/understanding-random-forest-58381e0602d2
#https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
##Project 4:
#p.show_graph(format='png')

from IPython.display import Image
from sklearn.tree import export_graphviz
from zipline.assets._assets import Equity  # Required for USEquityPricing
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.classifiers import Classifier
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.numpy_utils import int64_dtype


import alphalens as al
import graphviz
import numpy as np
import pandas as pd


from tqdm import tqdm

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)






class PricingLoader(object):
    def __init__(self, bundle_data):
        self.loader = USEquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader)

    def get_loader(self, column):
        if column not in USEquityPricing.columns:
            raise Exception('Column not in USEquityPricing')
        return self.loader



class Sector(Classifier):
    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1

    def __init__(self):
        #self.data = np.load('../../data/project_7_sector/data.npy')
        self.data = np.load('../../Project_7/data/data/project_7_sector/data.npy')

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )


def build_pipeline_engine(bundle_data, trading_calendar):
    pricing_loader = PricingLoader(bundle_data)

    engine = SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        calendar=trading_calendar.all_sessions,
        asset_finder=bundle_data.asset_finder)

    return engine



def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))


def plot(xs, ys, labels, title='', x_label='', y_label=''):
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()


def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    for x_train_i in range(len(importances)):
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))





def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns


def plot_factor_returns(factor_returns):
    (1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))

def sharpe_ratio_calc(factor_returns, annualization_factor=np.sqrt(252)):
    return annualization_factor * factor_returns.mean() / factor_returns.std()

def plot_factor_rank_autocorrelation(factor_data):
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))


def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[1]) #period = 1 1 day forward return to predict next day price using last 5 days data on which model is trained
        for factor_name, data in factor_data.iteritems()}




##############################################################################
#############  created a end of day data bundle for this project ##########
##############################################################################

import os
from pathlib import Path
import numpy as np
import pandas as pd
import zipline

from zipline.data import bundles

EOD_BUNDLE_NAME = 'eod-quotemedia'

#os.environ['ZIPLINE_ROOT'] = os.path.join(os.getcwd(), '..', '..', 'data', 'project_7_eod')

os.environ['ZIPLINE_ROOT'] = os.path.join(os.getcwd(), '..', '..', "Project_7/data/data/project_7_eod", "")


ingest_func = bundles.csvdir.csvdir_equities(['daily'], EOD_BUNDLE_NAME)
bundles.register(EOD_BUNDLE_NAME, ingest_func)


##############################################################################
#############  Build Pipeline Engine ##########
##############################################################################
'''Zipline's pipeline package to access data for this project.
 build a pipeline engine '''

from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume
from zipline.utils.calendars import get_calendar


universe = AverageDollarVolume(window_length=120).top(500) 
trading_calendar = get_calendar('NYSE') 
bundle_data = bundles.load(EOD_BUNDLE_NAME)
engine = build_pipeline_engine(bundle_data, trading_calendar)



##############################################################################
#############  View Data ##########
##############################################################################
'''With the pipeline engine built, get the stocks at the end
 of the period in the universe used '''

universe_end_date = pd.Timestamp('2016-01-05', tz='UTC')

universe_tickers = engine\
    .run_pipeline(
        Pipeline(screen=universe),
        universe_end_date,
        universe_end_date)\
    .index.get_level_values(1)\
    .values.tolist()
    
universe_tickers



##############################################################################
#############  Get Returns ##########
##############################################################################
'''access the returns data.start by building a data portal '''



from zipline.data.data_portal import DataPortal


data_portal = DataPortal(
    bundle_data.asset_finder,
    trading_calendar=trading_calendar,
    first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
    equity_minute_reader=None,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader)

''' built the helper function get_pricing to get the pricing from the data portal'''


def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field='close'):
    end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')
    start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC', offset='C')

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily')




##############################################################################
#############  Alpha Factors ##########
##############################################################################
'''use the following factors

Momentum 1 Year Factor
Mean Reversion 5 Day Sector Neutral Smoothed Factor
Overnight Sentiment Smoothed Factor  '''


from zipline.pipeline.factors import CustomFactor, DailyReturns, Returns, SimpleMovingAverage, AnnualizedVolatility
from zipline.pipeline.data import USEquityPricing

#type(DailyReturns)
#DailyReturns.dtype
#type(Returns)
#Returns.dtype

'''
see what some of the factor data looks like. For calculating factors, 
look back 3 years.Note: Going back 3 years falls on a day when the market is closed.
 Pipeline package doesn't handle start or end dates that don't fall on days
 when the market is not open. To fix this, we went back 2 extra days to fall on
 the next day when the market is open.
'''

factor_start_date = universe_end_date - pd.DateOffset(years=3, days=2)
sector = Sector()

def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()

def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    unsmoothed_factor = -Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=window_length) \
        .rank() \
        .zscore()

class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]
    
    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]
        
class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """
    window_safe = True
    
    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)

def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    unsmoothed_factor = TrailingOvernightReturns(inputs=[cto_out], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()




##############################################################################
#############  Combine the Factors to a single Pipeline ##########
##############################################################################
'''add the factors to a pipeline  '''

universe = AverageDollarVolume(window_length=120).top(500)
sector = Sector()

pipeline = Pipeline(screen=universe)
pipeline.add(
    momentum_1yr(252, universe, sector),
    'Momentum_1YR')
pipeline.add(
    mean_reversion_5day_sector_neutral_smoothed(20, universe, sector),
    'Mean_Reversion_Sector_Neutral_Smoothed')
pipeline.add(
    overnight_sentiment_smoothed(2, 10, universe),
    'Overnight_Sentiment_Smoothed')


##############################################################################
#############  Features and Labels ##########
##############################################################################
'''created some features that will help the model make predictions  '''

## "Universal" Quant Features

'''
Stock Volatility 20d, 120d
Stock Dollar Volume 20d, 120d
Sector
'''

pipeline.add(AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(), 'volatility_20d')
pipeline.add(AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(), 'volatility_120d')
pipeline.add(AverageDollarVolume(window_length=20, mask=universe).rank().zscore(), 'adv_20d')
pipeline.add(AverageDollarVolume(window_length=120, mask=universe).rank().zscore(), 'adv_120d')
pipeline.add(sector, 'sector_code')


## "Regime" Features

'''
try to capture market-wide regimes
High and low volatility 20d, 120d
High and low dispersion 20d, 120d
'''

class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns are days in rows, assets across columns
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns))**2))


pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20), 'dispersion_20d')
pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=120), 'dispersion_120d')

class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True
    
    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)
        out[:] = np.sqrt(260.* np.nanmean((mkt_returns-np.nanmean(mkt_returns))**2))


pipeline.add(MarketVolatility(window_length=20), 'market_vol_20d')
pipeline.add(MarketVolatility(window_length=120), 'market_vol_120d')



##############################################################################
#############  Target ##########
##############################################################################
'''try to predict the go forward 1-week return. 
When doing this, it's important to quantize the target, so make target market 
neutral and also normalize it to changing volatility and dispersion over time
this helps in making target robust to changes in market regime 
The factor created is the trailing 5-day return '''


pipeline.add(Returns(window_length=5, mask=universe).quantiles(2), 'return_5d')

'''
pipeline.add(Returns(window_length=2, mask=universe), 'daily_return_2')
pipeline.add(Returns(window_length=3, mask=universe), 'daily_return_3')
pipeline.add(Returns(window_length=4, mask=universe), 'daily_return_4')
all_factors_sub_set = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)
all_factors_sub_set[['daily_return_2','daily_return_3', 'daily_return_4']].reset_index().sort_values(['level_1', 'level_0']).head(50)
all_factors.head()
'''

''' sig'''
pipeline.add(Returns(window_length=5, mask=universe).quantiles(25), 'return_5d_p')



##############################################################################
#############  Date Features ##########
##############################################################################
'''make columns for the trees to split on
 that might capture trader/investor behavior due to calendar anomalies  '''

all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

all_factors['is_Janaury'] = all_factors.index.get_level_values(0).month == 1
all_factors['is_December'] = all_factors.index.get_level_values(0).month == 12
all_factors['weekday'] = all_factors.index.get_level_values(0).weekday
all_factors['quarter'] = all_factors.index.get_level_values(0).quarter
all_factors['qtr_yr'] = all_factors.quarter.astype('str') + '_' + all_factors.index.get_level_values(0).year.astype('str')
all_factors['month_end'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BM'))
all_factors['month_start'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BMS'))
all_factors['qtr_end'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BQ'))
all_factors['qtr_start'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BQS'))

all_factors.head(10)



##############################################################################
#############  One Hot Encode Sectors ##########
##############################################################################
'''For the model to better understand the sector data, 
one hot encode this data  '''



sector_lookup = pd.read_csv(
    os.path.join(os.getcwd(), '..', '..', 'Project_7/data/data/', 'project_7_sector', 'labels.csv'),
    index_col='Sector_i')['Sector'].to_dict()
sector_lookup

sector_columns = []
for sector_i, sector_name in sector_lookup.items():
    secotr_column = 'sector_{}'.format(sector_name)
    sector_columns.append(secotr_column)
    all_factors[secotr_column] = (all_factors['sector_code'] == sector_i)

all_factors[sector_columns].head()




##############################################################################
#############  Shift Target ##########
##############################################################################
'''use shifted 5 day returns for training the model  '''

all_factors['target'] = all_factors.groupby(level=1)['return_5d'].shift(-5)

all_factors[['return_5d','target']].reset_index().sort_values(['level_1', 'level_0']).head(780)



##############################################################################
#############  IID Check of Target ##########
##############################################################################
'''see if the returns are independent and identically distributed  '''

from scipy.stats import spearmanr


def sp(group, col1_name, col2_name):
    x = group[col1_name]
    y = group[col2_name]
    return spearmanr(x, y)[0]

''' sig '''
all_factors['target_5'] = all_factors.groupby(level=1)['return_5d'].shift(-5)

all_factors['target_1'] = all_factors.groupby(level=1)['return_5d'].shift(-4)
all_factors['target_2'] = all_factors.groupby(level=1)['return_5d'].shift(-3)
all_factors['target_3'] = all_factors.groupby(level=1)['return_5d'].shift(-2)
all_factors['target_4'] = all_factors.groupby(level=1)['return_5d'].shift(-1)

g = all_factors.dropna().groupby(level=0)
for i in range(5):
    label = 'target_'+str(i+1)
    ic = g.apply(sp, 'target', label)
    ic.plot(ylim=(-1, 1), label=label)
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.title('Rolling Autocorrelation of Labels Shifted 1,2,3,4 Days')
plt.show()


'''  
spearman rank corelation of returns varies from -0.4 to +0.8 and 
from graph seems like retruns are neither idependent nor 
identically distributed. Basically 
1> today's label is highly corelated to tommorows label 
2> today's label is highly corelated to day after tommorows label etc
'''


##############################################################################
#############  Train/Valid/Test Splits ##########
##############################################################################
'''split the data into a train, validation, and test dataset 

When splitting, make sure the data is in order from 
train, validation, and test respectivly. 
Say train_size is 0.7, valid_size is 0.2, and test_size is 0.1
first 70 percent of all_x and all_y would be the train set
next 20 percent of all_x and all_y would be the validation set
last 10 percent of all_x and all_y would be the test set. 
Esured not split a day between multiple datasets and 
contained within a single dataset
 '''



def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size == 1.0
    
    # TODO: Implement
    
    indexes = all_x.index.levels[0]
    train_idx, valid_idx, test_idx = np.split(indexes, [int(.6*len(indexes)), int(.8*len(indexes))])
    x_train, x_valid, x_test = all_x.ix[train_idx], all_x.ix[valid_idx], all_x.ix[test_idx ]
    y_train, y_valid, y_test = all_y.ix[train_idx], all_y.ix[valid_idx], all_y.ix[test_idx]
    return x_train,x_valid,x_test,y_train,y_valid,y_test

#####  use some of the features and the 5 day returns for our target



features = [
    'Mean_Reversion_Sector_Neutral_Smoothed', 'Momentum_1YR',
    'Overnight_Sentiment_Smoothed', 'adv_120d', 'adv_20d',
    'dispersion_120d', 'dispersion_20d', 'market_vol_120d',
    'market_vol_20d', 'volatility_20d',
    'is_Janaury', 'is_December', 'weekday',
    'month_end', 'month_start', 'qtr_end', 'qtr_start'] + sector_columns





'''
        
features_try = ['Mean_Reversion_Sector_Neutral_Smoothed']
all_factors.head() 
list(all_factors)
type(all_factors)
all_factors.dtypes
all_factors.index.levels[0]
all_factors.index.get_level_values(1)


all_factors_try =  all_factors['Mean_Reversion_Sector_Neutral_Smoothed']   
  

all_factors_try.head()
list(all_factors_try)
type(all_factors_try)
all_factors_try.dtypes
all_factors_try.index.levels[0]
all_factors_try.index.get_level_values(1)     
all_factors_try1 = all_factors_try.to_frame()
list(all_factors_try1)
type(all_factors_try1)
all_factors_try1.dtypes
all_factors_try1.index.levels[0]
all_factors_try1.index.get_level_values(1)     


all_factors_try2 =  all_factors['target'] 
list(all_factors_try2)
type(all_factors_try2)
all_factors_try2.dtypes
all_factors_try2.index.levels[0]
all_factors_try2.index.get_level_values(1) 
all_factors_try21 = all_factors_try2.to_frame()
list(all_factors_try21)
type(all_factors_try21)
all_factors_try21.dtypes
all_factors_try21.index.levels[0]
all_factors_try21.index.get_level_values(1)     


all_factors_final_try = pd.concat([all_factors_try1, all_factors_try21], axis=1)
list(all_factors_final_try)
type(all_factors_final_try)
all_factors_final_try.dtypes
all_factors_final_try.index.levels[0]
all_factors_final_try.index.get_level_values(1)     







temp_try_final = all_factors_final_try.dropna().copy()
X = temp_try_final[features_try]
y = temp_try_final[target_label]

'''

target_label = 'target'

temp = all_factors.dropna().copy()
X = temp[features]
y = temp[target_label]


X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y, 0.6, 0.2, 0.2)



##############################################################################
#############  Random Forests ##########
##############################################################################
'''single tree using our data'''



from IPython.display import display
from sklearn.tree import DecisionTreeClassifier


# This is to get consistent results between each run.
clf_random_state = 0

simple_clf = DecisionTreeClassifier(
    max_depth=3,
    criterion='entropy',
    random_state=clf_random_state)
simple_clf.fit(X_train, y_train)

'''
display(plot_tree_classifier(simple_clf, feature_names=features_try))
rank_features_by_importance(simple_clf.feature_importances_, features_try)
'''

display(plot_tree_classifier(simple_clf, feature_names=features))
rank_features_by_importance(simple_clf.feature_importances_, features)

'''
observe for information gain and Gini impurity
'''

##############################################################################
#############  Train Random Forests with Different Tree Sizes ##########
##############################################################################
'''build models using different tree sizes to find the model that best generalizes'''


##Using following Parameters

n_days = 10
n_stocks = 500

clf_parameters = {
    'criterion': 'entropy',
    'min_samples_leaf': n_stocks * n_days,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': clf_random_state}
n_trees_l = [50, 100, 250, 500, 1000]

''' 
choose a min_samples_leaf parameter to be small enough to allow the 
tree to fit the data with as much detail as possible, 
but not so much that it overfits. first tried with 500,
 which is the number of assets in the estimation universe.
 Since about 500 stocks in the stock universe, 
 want at least 500 stocks in a leaf for the leaf to make a 
 prediction that is representative. It’s common to multiply
 this by 2,3,5 or 10, so we’d have min samples
 leaf of 500, 1000, 1500, 2500, and 5000. 
 tried these values, notice that the model is “too good to be true” 
 on the training data. A good rule of thumb for what is
 considered “too good to be true”, and therefore a sign of 
 overfitting, is if the sharpe ratio is greater than 4. 
 Based on this, finally using min_sampes_leaf of 10 * 500, or 5,000

'''


'''
tried few other values for these parameters, but also keep in mind 
that making too many small adjustments to hyper-parameters
 can lead to overfitting even the validation data, and 
 therefore lead to less generalizable performance on the 
 out-of-sample test set. So when trying different parameter values, 
 choose values that are different enough in 
 scale (i.e. 10, 20, 100 instead of 10,11,12)
'''



from sklearn.ensemble import RandomForestClassifier    


train_score = []
valid_score = []
oob_score = []
feature_importances = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    clf.fit(X_train, y_train)
    
    train_score.append(clf.score(X_train, y_train.values))
    valid_score.append(clf.score(X_valid, y_valid.values))
    oob_score.append(clf.oob_score_)
    feature_importances.append(clf.feature_importances_)


#look at the accuracy of the classifiers over the number of trees

plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')


'''
accuracy is not great in this process and 
this leads the model to overfit, as Train score accuracy ~ 54.5%
 and OOB score accuracy ~ 53.5% are quite wide compared to valid score accuracy ~50.5%
'''
'''
when increasing the number of trees(increasing the complexity of model), 
see that the difference of accuracy between training set and validation set 
is sort of increasing, that's when it's called over-fitting.
'''

##average feature importance of the classifiers

print('Features Ranked by Average Importance:\n')
rank_features_by_importance(np.average(feature_importances, axis=0), features)

'''
some of the features of low to no importance. 
so removing them when training the final model
'''




##############################################################################
#############  Model Results ##########
##############################################################################
'''some additional metrics to see how well a model performs. 
 the function show_sample_results to show the following results of a model:

Sharpe Ratios
Factor Returns
Factor Rank Autocorrelation'''




import alphalens as al


all_assets = all_factors.index.levels[1].values.tolist()
all_pricing = get_pricing(
    data_portal,
    trading_calendar,
    all_assets,
    factor_start_date,
    universe_end_date)






##############################################################################
#############  Results ##########
##############################################################################
'''compare AI Alpha factor to a few other factors'''
'''
factor_names = [
    'Mean_Reversion_Sector_Neutral_Smoothed',
    'Momentum_1YR',
    'Overnight_Sentiment_Smoothed',
    'adv_120d',
    'volatility_20d']
'''
factor_names = [
    'Mean_Reversion_Sector_Neutral_Smoothed',
   ]


prob_array=[-1,1]
alpha_score = clf.predict_proba(X_test).dot(np.array(prob_array))

# Add Alpha Score to rest of the factors
alpha_score_label = 'AI_ALPHA'
factors_with_alpha = all_factors.loc[X_test.index].copy()
factors_with_alpha.head()
factors_with_alpha[alpha_score_label] = alpha_score


def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[1]) #period = 1 1 day forward return to predict next day price using last 5 days data on which model is trained
        for factor_name, data in factor_data.iteritems()}

'''
def build_factor_data_mutiple_period(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing) #period = 1 1 day forward return to predict next day price using last 5 days data on which model is trained
        for factor_name, data in factor_data.iteritems()}
'''

'''
A MultiIndex DataFrame indexed by date (level 0) and asset (level 1), 
containing the 
col1 --> values for a single alpha factor, 
col2 --> forward returns for each period (1d), 
col3 --> the factor quantile/bin that factor value belongs to,
col4 --> and (optionally) the group the asset belongs to
'''

factor_data = build_factor_data(factors_with_alpha[factor_names + [alpha_score_label]], all_pricing)
type(factor_data)


signal_try = factors_with_alpha[alpha_score_label]
signal_try_unstack = signal_try.unstack(level=1)
signal_try_unstack.sum(axis=1)

#factor_data_multi_period = build_factor_data(factors_with_alpha[factor_names + [alpha_score_label]], all_pricing)


#from alphalens.tears import create_full_tear_sheet
#create_full_tear_sheet(factor_data)


cleaned_smooth_factor = factor_data['AI_ALPHA']
type (cleaned_smooth_factor)

cleaned_smooth_factor.head()
cleaned_smooth_factor.tail()


signal_more_probable = cleaned_smooth_factor['factor']
signal_more_probable_unstack = signal_more_probable.unstack(level=1)

#cross check
signal_more_probable_unstack.sum(axis=1)



fact_weights = al.performance.factor_weights(cleaned_smooth_factor, demeaned=True, group_adjust=False, equal_weight=False)
fact_weights_unstack = fact_weights.unstack(level=1)
fact_weights_unstack.sum(axis=1)


























ls_factor_returns = pd.DataFrame()
for factor, factor_data in factor_data.items():
    ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

ls_factor_returns.head()
(1+ls_factor_returns).cumprod().plot()



from alphalens.tears import create_information_tear_sheet

create_information_tear_sheet(factor_data)

from alphalens.tears import create_returns_tear_sheet

create_returns_tear_sheet(factor_data)


from scipy.stats import zscore




























































unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}




































def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[1]) #period = 1 1 day forward return to predict next day price using last 5 days data on which model is trained
        for factor_name, data in factor_data.iteritems()}


def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns


def sharpe_ratio_calc(factor_returns, annualization_factor=np.sqrt(252)):
    return annualization_factor * factor_returns.mean() / factor_returns.std()

def plot_factor_returns(factor_returns):
    (1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))


def plot_factor_rank_autocorrelation(factor_data):
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))





def show_sample_results(data, samples, classifier, factors, pricing=all_pricing):
    # Calculate the Alpha Score
    prob_array=[-1,1]
    alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))
    
    # Add Alpha Score to rest of the factors
    alpha_score_label = 'AI_ALPHA'
    factors_with_alpha = data.loc[samples.index].copy()
    factors_with_alpha[alpha_score_label] = alpha_score
    
    # Setup data for AlphaLens
    print('Cleaning Data...\n')
    factor_data = build_factor_data(factors_with_alpha[factors + [alpha_score_label]], pricing)
    print('\n-----------------------\n')
    
    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(factor_data)
    sharpe_ratio = sharpe_ratio_calc(factor_returns)
    
    # Show Results
    print('             Sharpe Ratios')
    print(sharpe_ratio.round(2))
    plot_factor_returns(factor_returns)
    plot_factor_rank_autocorrelation(factor_data)


























































##############################################################################
#############  Training prediction - take 3-5 minutes, look for memory  ##########
##############################################################################
'''see how well the model runs on training data'''


show_sample_results(all_factors, X_train, clf, factor_names)



##############################################################################
#############  Validation prediction ##########
##############################################################################
'''see how well the model runs on validation data'''


show_sample_results(all_factors, X_valid, clf, factor_names)



'''
pretty extraordinary. Even when the input factor returns 
are sideways to down, the AI Alpha is positive with 
Sharpe Ratio close to 2. In order for this model to
perform well in production, need to correct though 
for the non-IID labels and mitigate likely overfitting
'''


##############################################################################
#############  Overlapping Samples - returns were not independent and identically distributed ##########
##############################################################################
'''fix this by removing overlapping samples.can be done as either of:

Don't use overlapping samples
Use BaggingClassifier's max_samples
Build an ensemble of non-overlapping trees
In this project, do all three methods and compare.'''



##############################################################################
#############  1. Drop Overlapping Samples ##########
##############################################################################
'''simplest of the three. just drop any overlapping 
samples from the dataset. Implement the 
non_overlapping_samples function to return a new dataset 
without overlapping samples'''


def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data
    
    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    
    # TODO: Implement
   

    non_overlapping_date_index = x.index.levels[0].tolist()[start_i :: n_skip_samples + 1]    
    return x.loc[non_overlapping_date_index], y.loc[non_overlapping_date_index]


'''

def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data
    
    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    
    # TODO: Implement
    
    #print(start_i)
    #print(x)
    len_x_ind = len(x.index.levels[0])
    #print(len_x_ind)
    
    
    
    
       
    #subx2 = x[::x.index.levels[0][2]]
    
    #keep = []
    #ind_drp = []
    #drop = 0
    #keep = start_i
    #while (keep<(len_x_ind-1)):
        
     #   keep = keep+n_skip_samples+1
        
     #   print (keep)
    
    step =3
    keep = []
    
    for i in range (0,len(x.index.levels[0]), step):
        keep.append(i+start_i)
        #print (keep)
    
    tryset = []
    for i in range (0,len(x.index.levels[0])):
        tryset.append (i)
        #print (tryset)
    
    drop = list(set(tryset) - set(keep))
    #print ('drop',drop)
    
    
    
    subx_set = []
    for d in drop:
        subx_set.append(x.index.levels[0][d])
        #print(subx_set)
    
    subx = x.drop(subx_set)
    #subx = x.take([x.index.levels[0][1],x.index.levels[0][4],x.index.levels[0][7]])
    
    
    #subx = x.drop([x.index.levels[0][0],x.index.levels[0][2],x.index.levels[0][3],x.index.levels[0][5],x.index.levels[0][6]])
    #print (subx)
    
    suby_set = []
    for d in drop:
        suby_set.append(y.index.levels[0][d])
        #print(suby_set)
    
    suby = y.drop(suby_set)
    #print (suby)
    
    
    #suby = y.drop([y.index.levels[0][0],y.index.levels[0][2],y.index.levels[0][3],y.index.levels[0][5],y.index.levels[0][6]])
    
    #ind_drop = []
    #for ind in x.index:
    #    if (ind) % n_skip_samples == 0:
    #        ind_drop.append(ind)

    #index_to_keep = set(range(x.shape[0])) - set(ind_drop)
    #xsub = x.take(list(index_to_keep))
    
    return subx, suby
'''

##############################################################################
#############  Train model - takes 2-3 minutes, look for memory ##########
##############################################################################

train_score = []
valid_score = []
oob_score = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    clf.fit(*non_overlapping_samples(X_train, y_train, 4))
    
    train_score.append(clf.score(X_train, y_train.values))
    valid_score.append(clf.score(X_valid, y_valid.values))
    oob_score.append(clf.oob_score_)

##Result

plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')




show_sample_results(all_factors, X_valid, clf, factor_names)


'''looks better, train accuracy ~ 51.8, oob ~ 51.1, valid ~ 50.6
but throwing away a lot of information by taking every 5th row'''



##############################################################################
############# 2. Use BaggingClassifier's max_samples ##########
##############################################################################

'''In this method, set max_samples to be on the order of 
the average uniqueness of the labels. Since  
RandomForrestClassifier does not take this param,
use BaggingClassifier. Implement bagging_classifier 
to build the bagging classifier'''

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def bagging_classifier(n_estimators, max_samples, max_features, parameters):
    """
    Build the bagging classifier.

    Parameters
    ----------
    n_estimators : int 
        The number of base estimators in the ensemble
    max_samples : float 
        The proportion of input samples drawn from when training each base estimator
    max_features : float 
        The proportion of input sample features drawn from when training each base estimator
    parameters : dict
        Parameters to use in building the bagging classifier
        It should contain the following parameters:
            criterion
            min_samples_leaf
            oob_score
            n_jobs
            random_state
    
    Returns
    -------
    bagging_clf : Scikit-Learn BaggingClassifier
        The bagging classifier
    """
    
    required_parameters = {'criterion', 'min_samples_leaf', 'n_jobs', 'random_state'}
    assert not required_parameters - set(parameters.keys())
    
    
    
    # TODO: Implement
    
    clf = BaggingClassifier(
        #base_estimator = base_estimator,
        n_estimators = n_estimators,
        max_samples = max_samples,
        max_features = max_features,
        oob_score=True,
        bootstrap=True,
        verbose=0,
        n_jobs=-1,
        random_state=0,
        base_estimator = DecisionTreeClassifier(
            criterion = parameters['criterion'],
            min_samples_leaf = parameters['min_samples_leaf'],
        #oob_score=parameters['oob_score'],
            #n_jobs = parameters['n_jobs'],
            #random_state=parameters['random_state'])
        #base_estimator = DecisionTreeClassifier(**parameters)
        )
        
    )
    clf.fit(X_train, y_train)
    return clf


##With the bagging classifier built,train a new model 
##and look at the results.


##############################################################################
#############  Train model - takes 35-40 minutes, look for memory ##########
##############################################################################


train_score = []
valid_score = []
oob_score = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = bagging_classifier(n_trees, 0.2, 1.0, clf_parameters)
    clf.fit(X_train, y_train)
    
    train_score.append(clf.score(X_train, y_train.values))
    valid_score.append(clf.score(X_valid, y_valid.values))
    oob_score.append(clf.oob_score_)


##Results
plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')


show_sample_results(all_factors, X_valid, clf, factor_names)

##seems much "better" in the sense that much better fidelity between the three



##############################################################################
#############  Build an ensemble of non-overlapping trees ##########
##############################################################################

'''
last method is to create ensemble of non-overlapping trees. 
Here writing a custom scikit-learn estimator. 
inherit from VotingClassifier and override the 
fit method so that it can be fit on non-overlapping periods
'''



import abc

from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError
        
    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError
    
    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf'+str(i), estimator) for i in range(n_skip_samples + 1)]
        
        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting)
    
    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)
        
        return self
    
    
## implement two abtract functions below
    
'''OOB Score
In order to get the correct OOB score, 
need to take the average of all the estimator's OOB scores.
Implemented calculate_oob_score to calculate this score.'''


    
def calculate_oob_score(classifiers):
    """
    Calculate the mean out-of-bag score from the classifiers.

    Parameters
    ----------
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to calculate the mean out-of-bag score
    
    Returns
    -------
    oob_score : float
        The mean out-of-bag score
    """
    
    # TODO: Implement
    
    oob = 0
    for clf in classifiers:
        oob = oob + clf.oob_score_
    return oob / len(classifiers)



'''Non Overlapping Estimators
calculate_oob_score implemented done, now create non overlapping
 estimators. Implement non_overlapping_estimators to build 
 non overlapping subsets of the data, then run a estimator 
 on each subset of data'''


def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to fit on the non overlapping data
    n_skip_samples : int
        The number of samples to skip
    
    Returns
    -------
    fit_classifiers : list of Scikit-Learn Classifiers
        The classifiers fit to the the non overlapping data
    """
    
    # TODO: Implement

    #print (classifiers)
    #print (type(classifiers)) 

    fit_classifiers = []
    for clf in classifiers:
        xx, yy = non_overlapping_samples(x, y, n_skip_samples, start_i=0)
        fit_classifiers.append(clf.fit(xx, yy))
    
    return fit_classifiers

class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)
        
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)



##Train Model - takes few mins only


train_score = []
valid_score = []
oob_score = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    
    clf_nov = NoOverlapVoter(clf)
    clf_nov.fit(X_train, y_train)
    
    train_score.append(clf_nov.score(X_train, y_train.values))
    valid_score.append(clf_nov.score(X_valid, y_valid.values))
    oob_score.append(clf_nov.oob_score_)


##Results


plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')

show_sample_results(all_factors, X_valid, clf_nov, factor_names)



##############################################################################
#############  Final Model -  Re-Training Model ##########
##############################################################################

'''
In production,roll forward the training. Typically re-train 
up to the "current day" and then test. for now I am train on 
the train & validation dataset.
'''



n_trees = 500

clf = RandomForestClassifier(n_trees, **clf_parameters)
clf_nov = NoOverlapVoter(clf)
clf_nov.fit(
    pd.concat([X_train, X_valid]),
    pd.concat([y_train, y_valid]))



##Results

#Accuracy
print('train: {}, oob: {}, valid: {}'.format(
    clf_nov.score(X_train, y_train.values),
    clf_nov.score(X_valid, y_valid.values),
    clf_nov.oob_score_))

#Train
show_sample_results(all_factors, X_train, clf_nov, factor_names)

#valids
show_sample_results(all_factors, X_valid, clf_nov, factor_names)

#Test
show_sample_results(all_factors, X_test, clf_nov, factor_names)

'''Despite the significant differences between the factor 
performances in the three sets, the AI APLHA is able to 
deliver positive performance.'''

'''
test set is considered as future data. Based on the graph on 
training/validation, AI_Alpha outperform other signals. That's 
 AI_Alpha instead of Momenum_1YR
'''





















































import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)


sample_df = pd.read_csv('C:/Users/Sudhin/horsepython/venv/Project_7/Sh.csv') 
sample_df

sample_df['date']

sample_df['date'] = pd.to_datetime(sample_df['date'])

#sample_df.set_index('date',drop=False, inplace=True)
#sample_df.index  = sample_df.index.tz_localize('GMT').tz_convert('US/Eastern')

#sample_df.index.tzinfo
sample_df['date']

sample_df.shape[1]

sample_df_add = sample_df.drop('lastupdated', axis=1)

import random

sample_df_add['me_rev_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['20d_vol_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['120d_vol_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['mom_1yr_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['20d_adv_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['120d_adv_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['disp_20d_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['disp_120d_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['over_sen_ran'] = np.random.uniform(-2,2,sample_df.shape[0])
sample_df_add['ret_5d'] = np.random.randint(0,2,sample_df.shape[0])




sample_factor = sample_df_add.drop(['open','volume','close','closeunadj','dividends','high','low'], axis=1)


sample_factor_all = sample_factor.groupby(['date', 'ticker']).nth(0)
sample_factor_all

sample_factor_all['target_ran'] = sample_factor_all.groupby(level=1)['ret_5d'].shift(-5)
sample_factor_all[['ret_5d','target_ran']].reset_index().sort_values(['level_1', 'level_0']).head(10)

from scipy.stats import spearmanr

def sp(group, col1_name, col2_name):
    x = group[col1_name]
    y = group[col2_name]
    return spearmanr(x,y)[0]

sample_factor_all['target_1'] = sample_factor_all.groupby(level=1)['ret_5d'].shift(-4)
sample_factor_all['target_2'] = sample_factor_all.groupby(level=1)['ret_5d'].shift(-3)
sample_factor_all['target_3'] = sample_factor_all.groupby(level=1)['ret_5d'].shift(-2)
sample_factor_all['target_4'] = sample_factor_all.groupby(level=1)['ret_5d'].shift(-1)




g = sample_factor_all.dropna().groupby(level=0)
for i in range(4):
    label = 'target_'+str(i+1)
    ic = g.apply(sp, 'target_ran', label)
    ic.plot(ylim=(-1, 1), label=label)
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.title('Rolling Autocorrelation of Labels Shifted 1,2,3,4 Days')
plt.show()


def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size == 1.0
    
    # TODO: Implement
    
    indexes = all_x.index.levels[0]
    train_idx, valid_idx, test_idx = np.split(indexes, [int(.6*len(indexes)), int(.8*len(indexes))])
    x_train, x_valid, x_test = all_x.ix[train_idx], all_x.ix[valid_idx], all_x.ix[test_idx ]
    y_train, y_valid, y_test = all_y.ix[train_idx], all_y.ix[valid_idx], all_y.ix[test_idx]
    return x_train,x_valid,x_test,y_train,y_valid,y_test


features = [
    'me_rev_ran', '20d_vol_ran',
    '120d_vol_ran', 'mom_1yr_ran', '20d_adv_ran',
    '120d_adv_ran', 'disp_20d_ran', 'disp_120d_ran',
    'over_sen_ran']

target_label = 'target_ran'

temp = sample_factor_all.dropna().copy()
X = temp[features]
y = temp[target_label]

X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y, 0.6, 0.2, 0.2)

X_train.head()
y_train.head()

from tqdm import tqdm

from IPython.display import display
from sklearn.tree import DecisionTreeClassifier


# This is to get consistent results between each run.
clf_random_state = 0

simple_clf = DecisionTreeClassifier(
    max_depth=3,
    criterion='entropy',
    random_state=clf_random_state)
simple_clf.fit(X_train, y_train)


n_days = 10
n_stocks = 500

clf_parameters = {
    'criterion': 'entropy',
    'min_samples_leaf': n_stocks * n_days,
    'oob_score': True,
    'n_jobs': -1,
    'random_state': clf_random_state}
n_trees_l = [50, 100, 250, 500, 1000]

from sklearn.ensemble import RandomForestClassifier


train_score = []
valid_score = []
oob_score = []
feature_importances = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    clf.fit(X_train, y_train)
    
    train_score.append(clf.score(X_train, y_train.values))
    valid_score.append(clf.score(X_valid, y_valid.values))
    oob_score.append(clf.oob_score_)
    feature_importances.append(clf.feature_importances_)


def plot(xs, ys, labels, title='', x_label='', y_label=''):
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.45, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()

plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees', 'scores')


all_assets = sample_factor_all.index.levels[1].values.tolist()

sample_df
all_pricing_df = sample_df.drop(['open','volume','closeunadj','dividends','high','low','lastupdated'], axis=1)
all_pricing = pd.pivot_table(all_pricing_df, values = 'close', index = ['date'], columns = ['ticker'] ) 



import alphalens as al

'''def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, periods=[1])
        for factor_name, data in factor_data.iteritems()}

def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, quantiles=None, bins=[-np.inf, np.inf])
        for factor_name, data in factor_data.iteritems()}'''

def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, quantiles=None, bins=1)
        for factor_name, data in factor_data.iteritems()}
            
            

def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:,0]
    
    return ls_factor_returns


def sharpe_ratio_calc(factor_returns,annulization_factor=np.sqrt(252)):
    return annulization_factor * factor_returns.mean()/factor_returns.std()

def plot_factor_returns(factor_returns):
    (1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))


def plot_factor_rank_autocorrelation(factor_data):
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))




def show_sample_results(data, samples, classifier, factors, pricing=all_pricing):
    # Calculate the Alpha Score
    prob_array=[-1,1]
    alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))
    
    # Add Alpha Score to rest of the factors
    alpha_score_label = 'AI_ALPHA'
    factors_with_alpha = data.loc[samples.index].copy()
    factors_with_alpha[alpha_score_label] = alpha_score
    
    # Setup data for AlphaLens
    print('Cleaning Data...\n')
    factor_data = build_factor_data(factors_with_alpha[factors + [alpha_score_label]], pricing)
    print('\n-----------------------\n')
    
    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(factor_data)
    sharpe_ratio = sharpe_ratio_calc(factor_returns)
    
    # Show Results
    print('             Sharpe Ratios')
    print(sharpe_ratio.round(2))
    plot_factor_returns(factor_returns)
    plot_factor_rank_autocorrelation(factor_data)
    


'''prob_array=[-1,1]  
alpha_score = clf.predict_proba(X_train).dot(np.array(prob_array))    
alpha_score_label = 'AI_ALPHA'
factors_with_alpha = sample_factor_all.loc[X_train.index].copy()
factors_with_alpha[alpha_score_label] = alpha_score
factor_data = build_factor_data(factors_with_alpha[factor_names + [alpha_score_label]], all_pricing)

type(factor_data)

factor_returns = get_factor_returns(factor_data)'''




    
factor_names = [
    'me_rev_ran',
    'mom_1yr_ran',
    'over_sen_ran',
    '120d_adv_ran',
    '20d_vol_ran']


sample_factor_all.index.levels[0]

show_sample_results(sample_factor_all, X_train, clf, factor_names)

show_sample_results(sample_factor_all, X_valid, clf, factor_names)




def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data
    
    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    
    # TODO: Implement
    
    #print(start_i)
    #print(x)
    len_x_ind = len(x.index.levels[0])
    #print(len_x_ind)
    
    
    
    
       
    #subx2 = x[::x.index.levels[0][2]]
    
    #keep = []
    #ind_drp = []
    #drop = 0
    #keep = start_i
    #while (keep<(len_x_ind-1)):
        
     #   keep = keep+n_skip_samples+1
        
     #   print (keep)
    
    step =3
    keep = []
    
    for i in range (0,len(x.index.levels[0]), step):
        keep.append(i+start_i)
        #print (keep)
    
    tryset = []
    for i in range (0,len(x.index.levels[0])):
        tryset.append (i)
        #print (tryset)
    
    drop = list(set(tryset) - set(keep))
    #print ('drop',drop)
    
    
    
    subx_set = []
    for d in drop:
        subx_set.append(x.index.levels[0][d])
        #print(subx_set)
    
    subx = x.drop(subx_set)
    #subx = x.take([x.index.levels[0][1],x.index.levels[0][4],x.index.levels[0][7]])
    
    
    #subx = x.drop([x.index.levels[0][0],x.index.levels[0][2],x.index.levels[0][3],x.index.levels[0][5],x.index.levels[0][6]])
    #print (subx)
    
    suby_set = []
    for d in drop:
        suby_set.append(y.index.levels[0][d])
        #print(suby_set)
    
    suby = y.drop(suby_set)
    #print (suby)
    
    
    #suby = y.drop([y.index.levels[0][0],y.index.levels[0][2],y.index.levels[0][3],y.index.levels[0][5],y.index.levels[0][6]])
    
    #ind_drop = []
    #for ind in x.index:
    #    if (ind) % n_skip_samples == 0:
    #        ind_drop.append(ind)

    #index_to_keep = set(range(x.shape[0])) - set(ind_drop)
    #xsub = x.take(list(index_to_keep))
    
    return subx, suby


train_score = []
valid_score = []
oob_score = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    clf.fit(*non_overlapping_samples(X_train, y_train, 4))
    
    train_score.append(clf.score(X_train, y_train.values))
    valid_score.append(clf.score(X_valid, y_valid.values))
    oob_score.append(clf.oob_score_)


plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees', 'scores')



show_sample_results(sample_factor_all, X_valid, clf, factor_names)



from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def bagging_classifier(n_estimators, max_samples, max_features, parameters):
    """
    Build the bagging classifier.

    Parameters
    ----------
    n_estimators : int 
        The number of base estimators in the ensemble
    max_samples : float 
        The proportion of input samples drawn from when training each base estimator
    max_features : float 
        The proportion of input sample features drawn from when training each base estimator
    parameters : dict
        Parameters to use in building the bagging classifier
        It should contain the following parameters:
            criterion
            min_samples_leaf
            oob_score
            n_jobs
            random_state
    
    Returns
    -------
    bagging_clf : Scikit-Learn BaggingClassifier
        The bagging classifier
    """
    
    required_parameters = {'criterion', 'min_samples_leaf', 'n_jobs', 'random_state'}
    assert not required_parameters - set(parameters.keys())
    
    
    
    # TODO: Implement
    
    clf = BaggingClassifier(
        #base_estimator = base_estimator,
        n_estimators = n_estimators,
        max_samples = max_samples,
        max_features = max_features,
        oob_score=True,
        bootstrap=True,
        verbose=0,
        n_jobs=-1,
        random_state=0,
        base_estimator = DecisionTreeClassifier(
            criterion = parameters['criterion'],
            min_samples_leaf = parameters['min_samples_leaf'],
        #oob_score=parameters['oob_score'],
            #n_jobs = parameters['n_jobs'],
            #random_state=parameters['random_state'])
        #base_estimator = DecisionTreeClassifier(**parameters)
        )
        
    )
    clf.fit(X_train, y_train)
    return clf


train_score = []
valid_score = []
oob_score = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = bagging_classifier(n_trees, 0.2, 1.0, clf_parameters)
    clf.fit(X_train, y_train)
    
    train_score.append(clf.score(X_train, y_train.values))
    valid_score.append(clf.score(X_valid, y_valid.values))
    oob_score.append(clf.oob_score_)


plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')


show_sample_results(sample_factor_all, X_valid, clf, factor_names)

import abc

from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError
        
    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError
    
    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf'+str(i), estimator) for i in range(n_skip_samples + 1)]
        
        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting)
    
    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)
        
        return self

def calculate_oob_score(classifiers):
    """
    Calculate the mean out-of-bag score from the classifiers.

    Parameters
    ----------
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to calculate the mean out-of-bag score
    
    Returns
    -------
    oob_score : float
        The mean out-of-bag score
    """
    
    # TODO: Implement
    
    oob = 0
    for clf in classifiers:
        oob = oob + clf.oob_score_
    return oob / len(classifiers)

def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    classifiers : list of Scikit-Learn Classifiers
        The classifiers used to fit on the non overlapping data
    n_skip_samples : int
        The number of samples to skip
    
    Returns
    -------
    fit_classifiers : list of Scikit-Learn Classifiers
        The classifiers fit to the the non overlapping data
    """
    
    # TODO: Implement

    #print (classifiers)
    #print (type(classifiers)) 

    fit_classifiers = []
    for clf in classifiers:
        xx, yy = non_overlapping_samples(x, y, n_skip_samples, start_i=0)
        fit_classifiers.append(clf.fit(xx, yy))
    
    return fit_classifiers

class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)
        
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)

train_score = []
valid_score = []
oob_score = []

for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    
    clf_nov = NoOverlapVoter(clf)
    clf_nov.fit(X_train, y_train)
    
    train_score.append(clf_nov.score(X_train, y_train.values))
    valid_score.append(clf_nov.score(X_valid, y_valid.values))
    oob_score.append(clf_nov.oob_score_)

plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')


plot(
    [n_trees_l]*3,
    [train_score, valid_score, oob_score],
    ['train', 'validation', 'oob'],
    'Random Forrest Accuracy',
    'Number of Trees')

show_sample_results(sample_factor_all, X_valid, clf_nov, factor_names)


n_trees = 500

clf = RandomForestClassifier(n_trees, **clf_parameters)
clf_nov = NoOverlapVoter(clf)
clf_nov.fit(
    pd.concat([X_train, X_valid]),
    pd.concat([y_train, y_valid]))

print('train: {}, oob: {}, valid: {}'.format(
    clf_nov.score(X_train, y_train.values),
    clf_nov.score(X_valid, y_valid.values),
    clf_nov.oob_score_))

show_sample_results(sample_factor_all, X_train, clf_nov, factor_names)

show_sample_results(sample_factor_all, X_valid, clf_nov, factor_names)

show_sample_results(sample_factor_all, X_test, clf_nov, factor_names)






























df2 =  pd.DataFrame(np.random.randint(3, 3), columns=['A', 'B', 'C'])
# randn(3,3) returns nine random numbers in a 3x3 array.
# the columns argument to DataFrame names the 3 columns. 
# no datetimes here! (look at df2 to check)

df2['A'] = pd.to_datetime(df2['A'])
# convert the random numbers to datetimes -- look at df2 again
# if A had values to_datetime couldn't handle, we'd clean up A first

df2.set_index('A',drop=False, inplace=True)
# and use that column as an index for the whole df2;

df2.index  = df2.index.tz_localize('GMT').tz_convert('US/Eastern')
# make it timezone-conscious in GMT and convert that to Eastern

df2.index.tzinfo