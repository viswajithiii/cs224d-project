import sys


def compute_overlap_confusion(preds, trues):
    """
    Given the true and predicted labels,
    computes the overlap P/R/F1 as done in the paper.
    Returns both proportional and binary.
    """
    data_len = len(preds)
    assert len(trues) == data_len

    # State can be None, DSE, ESE depending on whether
    # we are in a phrase or not.
    state = None

    curr_window_begin = None

    total_true_windows = {'DSE': 0, 'ESE': 0}
    correct_true_windows = {'DSE': 0, 'ESE': 0}  # For binary recall.

    total_true_windows_words = {'DSE': 0, 'ESE': 0}
    correct_true_windows_words = {'DSE': 0, 'ESE': 0}

    sum_true_proportions = {'DSE': 0., 'ESE': 0.}  # For proportional recall.

    # First, we iterate over the true labels, and compute
    # recall.
    for i in xrange(data_len):
        if state is None:
            if trues[i] == 'O':
                pass
            elif trues[i] == 'BDSE':
                curr_window_begin = i
                state = 'DSE'
            elif trues[i] == 'BESE':
                curr_window_begin = i
                state = 'ESE'
            else:
                assert False  # This shouldn't happen.

        elif state == 'DSE':

            # The window has ended. Now we can compute recall.
            if trues[i] == 'O' or trues[i].startswith('B'):
                total_true_windows['DSE'] += 1
                total_true_windows_words['DSE'] += (i - curr_window_begin)
                window_len = (i - curr_window_begin)
                num_correct = 0
                binary_overlap = False
                for j in xrange(curr_window_begin, i):
                    if preds[j].endswith('DSE'):
                        binary_overlap = True
                        num_correct += 1
                        correct_true_windows_words['DSE'] += 1
                if binary_overlap:
                    correct_true_windows['DSE'] += 1
                    sum_true_proportions['DSE'] += float(num_correct)/window_len
                curr_window_begin = None if trues[i] == 'O' else i
                state = None if trues[i] == 'O' else trues[i][1:]
            elif trues[i] == 'IDSE':
                pass
            else:
                assert False  # This shouldn't happen.

        elif state == 'ESE':
            if trues[i] == 'O' or trues[i].startswith('B'):
                total_true_windows['ESE'] += 1
                total_true_windows_words['ESE'] += (i - curr_window_begin)
                window_len = (i - curr_window_begin)
                num_correct = 0
                binary_overlap = False
                for j in xrange(curr_window_begin, i):
                    if preds[j].endswith('ESE'):
                        binary_overlap = True
                        correct_true_windows_words['ESE'] += 1
                        num_correct += 1
                if binary_overlap:
                    correct_true_windows['ESE'] += 1
                    sum_true_proportions['ESE'] += float(num_correct)/window_len
                curr_window_begin = None if trues[i] == 'O' else i
                state = None if trues[i] == 'O' else trues[i][1:]
            elif trues[i] == 'IESE':
                pass
            else:
                assert False  # This shouldn't happen.

    binary_recall = {}
    prop_recall = {}
    for s in ['DSE', 'ESE']:
        print 'For', s
        print 'Total true windows:', total_true_windows[s]
        print 'Total number of windows with some overlap:', correct_true_windows[s]
        binary_recall[s] = float(correct_true_windows[s])/total_true_windows[s]
        print 'Binary recall:', binary_recall[s]
        print 'Total number of words in true windows:', total_true_windows_words[s]
        print 'Total number of overlapping words:', correct_true_windows_words[s]
        print 'Sum true proportions:', sum_true_proportions[s]
        prop_recall[s] = float(sum_true_proportions[s])/total_true_windows[s]
        print 'Proportional recall:', prop_recall[s]


    #  Next, we iterate over the predicted labels and compute precision.
    total_pred_windows = {'DSE': 0, 'ESE': 0}
    correct_pred_windows = {'DSE': 0, 'ESE': 0}  # For binary prec.

    total_pred_windows_words = {'DSE': 0, 'ESE': 0}
    correct_pred_windows_words = {'DSE': 0, 'ESE': 0}  # For proportional prec.
    sum_pred_proportions = {'DSE': 0., 'ESE' : 0.}
    for i in xrange(data_len):
        if state is None:
            if preds[i] == 'O':
                pass
            elif preds[i] == 'BDSE':
                curr_window_begin = i
                state = 'DSE'
            elif preds[i] == 'BESE':
                curr_window_begin = i
                state = 'ESE'
            # We will be kind to our model; if it starts with an I
            # directly, we will interpret it as a B
            else:
                curr_window_begin = i
                state = preds[i][1:]

        elif state == 'DSE':

            # The window has ended. Now we can compute recall.
            if preds[i] == 'O' or preds[i].startswith('B') or preds[i] == 'IESE':
                total_pred_windows['DSE'] += 1
                total_pred_windows_words['DSE'] += (i - curr_window_begin)
                binary_overlap = False
                window_len = (i - curr_window_begin)
                num_correct = 0
                for j in xrange(curr_window_begin, i):
                    if trues[j].endswith('DSE'):
                        binary_overlap = True
                        correct_pred_windows_words['DSE'] += 1
                        num_correct += 1
                if binary_overlap:
                    correct_pred_windows['DSE'] += 1
                    sum_pred_proportions['DSE'] += float(num_correct)/window_len
                curr_window_begin = None if preds[i] == 'O' else i
                state = None if preds[i] == 'O' else preds[i][1:]
            elif preds[i] == 'IDSE':
                pass
            else:
                assert False  # This shouldn't happen.

        elif state == 'ESE':
            if preds[i] == 'O' or preds[i].startswith('B') or preds[i] == 'IDSE':
                total_pred_windows['ESE'] += 1
                total_pred_windows_words['ESE'] += (i - curr_window_begin)
                binary_overlap = False
                window_len = (i - curr_window_begin)
                num_correct = 0
                for j in xrange(curr_window_begin, i):
                    if trues[j].endswith('ESE'):
                        binary_overlap = True
                        correct_pred_windows_words['ESE'] += 1
                        num_correct += 1
                if binary_overlap:
                    correct_pred_windows['ESE'] += 1
                    sum_pred_proportions['ESE'] += float(num_correct)/window_len
                curr_window_begin = None if preds[i] == 'O' else i
                state = None if preds[i] == 'O' else preds[i][1:]
            elif preds[i] == 'IESE':
                pass
            else:
                assert False  # This shouldn't happen.

    binary_prec= {}
    prop_prec = {}
    for s in ['DSE', 'ESE']:
        print 'For', s
        print 'Total pred windows:', total_pred_windows[s]
        print 'Total number of windows with some overlap:', correct_pred_windows[s]
        binary_prec[s] = float(correct_pred_windows[s])/total_pred_windows[s]
        print 'Binary precision:', binary_prec[s]
        print 'Total number of words in pred windows:', total_pred_windows_words[s]
        print 'Total number of overlapping words:', correct_pred_windows_words[s]
        print 'Sum pred propotions:', sum_pred_proportions[s]
        prop_prec[s] = float(sum_pred_proportions[s])/total_pred_windows[s]
        print 'Proportional precision:', prop_prec[s]

    print
    print '#### FINAL SUMMARY ####'
    for s in ['DSE', 'ESE']:
        print 'For', s,':'
        print 'Binary:',
        print 'P {:2.4f} / R {:2.4f} / F1 {:2.4f}'.format(binary_prec[s], binary_recall[s], 2*binary_prec[s]*binary_recall[s]/(binary_prec[s] + binary_recall[s]))
        print 'Proportional:',
        print 'P {:2.4f} / R {:2.4f} / F1 {:2.4f}'.format(prop_prec[s], prop_recall[s], 2*prop_prec[s]*prop_recall[s]/(prop_prec[s] + prop_recall[s]))



def load_from_file(fname):

    f = open(fname, 'r')
    preds = []
    trues = []
    for line in f:
        sp_line = line.strip().split('\t')
        preds.append(sp_line[1])
        trues.append(sp_line[2])
    return preds, trues


fname = sys.argv[1]
preds, trues = load_from_file(fname)
compute_overlap_confusion(preds, trues)
