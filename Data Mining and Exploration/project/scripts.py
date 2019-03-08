import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

### read_transform_contentfeatures ###
# Use this code to load the "content based features" and convert to all numbers.
# CAREFUL: I'm assigning a dummy value here. It's probably better to fill with means for example
###
# INPUT:
#	file ./data/features/v2-DiscoveryChallenge2010.content_based_features.csv
# OUTPUT:
#	var X_content_full: pd.DataFrame with unified content features
#	var keys_content_features: np.ndarray(String) of feature columns
###

def read_transform_contentfeatures(file='./data/features/v2-DiscoveryChallenge2010.content_based_features.csv'):
	X_content_full = pd.read_csv(file)
	X_content_full.replace('?',-999,inplace=True) # assign DUMMY value to empty values
	X_content_full = X_content_full.convert_objects(convert_numeric=True) # now we can actually convert to numbers
	#X_content_full.describe().T

	keys_content_features = X_content_full.keys().drop(['#hostid', 'hostname'])
	
	return (X_content_full, keys_content_features)


### read_transform_labels ###
# Use this code to load the labels and unify the binary features.
# As the binary features are not always maintained, I chose the following mapping:
# YES = 1, NO = -1, ? | NaN = 0
###
# INPUT:
#	file ./data/labels/en.useful-labels.csv
# OUTPUT:
#	var y_full: pd.DataFrame with unified labels
#	var keys_labels: np.ndarray(String) of possible target columns
###

def read_transform_labels(file='./data/labels/en.useful-labels.csv'):
	y_full = pd.read_csv(file,sep=';')

	### unify labels

	# mapping e.g. column 'Adult Content' to values, e.g. 'Adult' and 'NonAdult'
	keymap = {
		'Adult Content': 'Adult',
		'Other Problem': 'OtherProblem',
		'Web Spam': 'Spam',
		'News/Editorial': 'News-Edit',
		'Commercial': 'Commercial',
		'Educational/Research': 'Educational',
		'Discussion': 'Discussion',
		'Personal/Leisure': 'Personal-Leisure',
		'Media': 'Media',
		'Database': 'Database',
	}

	for key in y_full.keys():
		if key in keymap.keys():
			key2 = keymap[key]
		else:
			key2 = key
			
		y_full[key].replace('Non{}'.format(key2),-1,inplace=True)
		y_full[key].replace('No{}'.format(key2),-1,inplace=True)
		y_full[key].replace(key2,1,inplace=True)
		y_full[key].replace('NaN',0,inplace=True)
		
	y_full['Confidence'].replace('Unsure'.format(key2),-1,inplace=True)
	y_full['Confidence'].replace('Sure',1,inplace=True)
	y_full['Confidence'].replace('nan',0,inplace=True)
		

	### remember "useful" keys (labels only, cut off admin data)
	keys_labels = y_full.keys().drop(['ID','UserID','Date', 'Hosting Type', 'Language','auto_lang'])

	#y_full.describe()
	
	return(y_full,keys_labels)
	
### join_data
# join the unified data together to get usable X-y pairs
# CAREFUL: This is probably wrong! I think the actual nature of the data
#          is more complex than this simple join on host IDs
##
# INPUT:
#	file ./data/labels/en.dns-response.csv (do we even have this for test data?)
#	X_content_full from read_transform_contentfeatures.py
#	y_full from read_transform_labels.py
# OUTPUT:
#	fulldata pd.DataFrame with input features, labels and administrative data
##


def join_data(X_content_full, y_full, file='./data/labels/en.dns-response.csv'):
	hostdata = pd.read_csv(file,sep=';')
	y_joined = pd.merge(hostdata,y_full,on='ID',how='inner') # Pandas inner join: host data & y
	y_joined.drop_duplicates(['ID'],inplace=True) # not sure about this
	fulldata = pd.merge(y_joined,X_content_full,left_on='ID',right_on='#hostid') # join X with previous join
	return(fulldata)

def print_run_time(t0):
    '''t0 is the start time, given by time.time()'''
    print 'run time: {:0.2f} min'.format((time.time() - t0)/60)
    

def tf_idf(tf, df, N):
    '''
    tf-idf (term frequency - inverse document frequency) calculation, which we'll use for the word embeddings.
    Source: http://www.inf.ed.ac.uk/teaching/courses/nlu/lectures/nlu_l02-vsm-2x2.pdf
    '''
    return (1 + np.log2(tf)) * np.log2(N / float(df))


def create_sparse_embeddings(filename, hostids=None, embedding_type='tf-idf'):
    '''
    Takes v2-host_tfdf.en.txt and outputs sparse 
    
    input:
        filename (str): file path to v2-host_tfdf.en.txt (originally in v2-all_in_one/features/tfdf/)
        hostids (list of int): only pull these hostids
        embedding_type (str): either 'tf-idf' or 'tf'

    output:
        dictionary with format {hostid1: [(wordid1, embedding_value), (wordid2, embedding_value), ...], hostid2: [(wordid7, embedding_value), ...]}
    '''
    
    # Read raw data from file.
    with open(filename, 'r') as f:
        raw_data = f.readlines()
    
    N = len(raw_data)  # N in tf-idf calculation
    sparse = {}  # initialize dictionary
    
    for line in raw_data:
        
        line = re.sub('\n', '', line)  # remove \n at end
        split_line = line.split(' ')
        
        # For structure of file, see: https://dms.sztaki.hu/node/350
        
        # First element is hostid.
        hostid = int(split_line.pop(0))
        
        if hostids and hostid not in hostids:
            continue
        
        sparse[hostid] = {}
        
        # We then have groups of three: wordid1 tf1 df1, wordid2 tf2 df2, ...
        while len(split_line) >= 3:
            
            # Remove first three elements.
            wordid, tf, df = [int(s) for s in split_line[:3]]
            del split_line[:3]
            
            # Add to dictionary.
            if embedding_type == 'tf-idf':
                sparse[hostid][wordid] = tf_idf(tf, df, N)
            elif embedding_type == 'tf':
                sparse[hostid][wordid] = tf
            else:
                raise ValueError('embedding_type needs to be \'tf-idf\' or \'tf\'')
            
    return sparse

# Plot confusion matrix by using seaborn heatmap function
def plot_confusion_matrix(cm, normalize=False, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix.
    
    If normalize is set to True, the rows of the confusion matrix are normalized so that they sum up to 1.
    
    """
    if normalize is True:
        cm = cm/cm.sum(axis=1)[:, np.newaxis]
        vmin, vmax = 0., 1.
        fmt = '.2f'
    else:
        vmin, vmax = None, None
        fmt = 'd'
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=vmin, vmax=vmax, 
                    annot=True, annot_kws={"fontsize":9}, fmt=fmt)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')