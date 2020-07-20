import matplotlib.pyplot as plt

def plot(X, idx, title):
    '''
    Show clustering results

    Input:  X: data point features, n-by-p maxtirx.
            idx: data point cluster labels, n-by-1 vector.
    '''
    plt.figure(figsize=(6, 6))
    # print(len(X[idx==(1, 1)]))
    plt.plot(X[idx==0, 0],X[idx==0, 1],'r.',markersize=5, label='Cluster 1')
    plt.plot(X[idx==1, 0],X[idx==1, 1],'b.', markersize=5, label='Cluster 2')
    plt.title(title)
    plt.legend(loc='upper right')
#     plt.show()
    plt.savefig('./Results_Plots/'+title + '.svg', format='svg')