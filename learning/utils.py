import os
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

COLORS1 = [
    (148, 255, 181), (66, 102, 0), (116, 10, 155), (94, 241, 242),
    (0, 153, 143), (0, 255, 211), (128, 128, 128), (143, 124, 0),
    (157, 204, 0), (194, 0, 136), (255, 164, 5), (255, 168, 187),  (255, 0, 16),
     (0, 153, 143), (224, 255, 102), (153, 0, 0), (255, 255, 128),
      (255, 255, 0), (255, 80, 5)
]


def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        mode='max', img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses)+1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[0].set_ylim(0.0, 2)
    axes[1].plot(np.arange(1, len(step_scores)+1), step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores)+1), eval_scores, color='r', marker='')
    if mode == 'max':
        axes[1].set_ylim(0.5, 1.0)
    else:    # mode == 'min'
        axes[1].set_ylim(0.0, 0.5)
    axes[1].set_ylabel('Pixel Accuracy')
    axes[1].set_xlabel('Number of epochs')

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)
    plt.close()

def draw_pixel(y_pred, threshold=None):
    color = COLORS1
    is_batch = len(y_pred.shape) == 4

    if not is_batch:
        y_pred = np.expand_dims(y_pred, axis=0)

    class_num = y_pred.shape[-1]
    if threshold is None:
        mask_pred = np.argmax(y_pred, axis=-1)
        mask_output = np.zeros(list(mask_pred.shape)+[3])
        for i in range(1, class_num):
            mask_output[np.where(mask_pred==i)] = color[i]
    else:
        mask_output = np.zeros(list(mask_pred.shape[:-1])+[3])
        for i in range(1, class_num):
            mask_output[np.where(mask_pred[...,i] > threshold)] = color[i]

    if is_batch:
        return mask_output
    else:
        return mask_output[0]
