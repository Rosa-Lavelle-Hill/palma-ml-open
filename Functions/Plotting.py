import matplotlib
import shap
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import base64
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
import colorcet as cc
import matplotlib.colors as colors
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm, Colormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import TransformedBbox

def plot_hist(save_name, x, bins, save_path, title, xlim=None, ylim=None,
              fig_size=(20,20), xlab="", ylab='', fontsize=10, html=False, save_name_exceptions=[]):
    """Plots a histogram and saves to file. Also outputs a html image tag if html=True"""
    if isinstance(x, np.ndarray) == True:
        x = pd.Series(x.reshape((x.shape[0],)))
    plt.figure(figsize=fig_size)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if (x.isnull().all() == False):
        plt.hist(x, bins = bins, color ="skyblue", alpha=0.5)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)
        if save_name in save_name_exceptions:
            print('odd case')
        elif x.dtype == "str":
            print('string')
        elif (x.dtype == "float64") or (x.dtype == "int"):

                plt.axvline(x=np.mean(x) - np.std(x), ls="--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=np.mean(x) + np.std(x), ls="--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=np.mean(x), ls="-", color='red', alpha=0.7)
                plt.axvline(x=np.percentile(x, 5), ls="dotted", color='lightblue', alpha=0.7)
                plt.axvline(x=np.percentile(x, 95), ls="dotted", color='lightblue', alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_path+save_name+".png")
                plt.clf()
                plt.cla()
                plt.close()
        else:
            print("not int or float")
    else:
            print("nas")
    if html==True:
            data_uri = base64.b64encode(open(save_path+save_name+".png", 'rb').read()).decode('utf-8')
            image_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
            return image_tag
    else:
        return



def plot_bar(x, y, hue, df,
             ylab, xlab, save_path, save_name,
             title=" ",
             orient='h',
             pallette="colorblind",
             fontsize=6,
             ):
    l = len(df[y].unique())
    pallette = sns.color_palette(pallette, l)
    plt.figure(figsize=(20, 60))
    chart = sns.catplot(x=y, y=x, hue=hue, data=df,
                        legend=False,
                        kind='bar',
                        orient=orient,
                        ci=None, palette=pallette)
    plt.title(title, fontsize=12)
    plt.xticks(fontsize=fontsize)
    chart.set(xlabel=xlab, ylabel=ylab)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_bar_h_df(x, y, df,
               ylab, xlab,
               save_path, save_name,
               title=" ",
               fontsize=16):
        x=np.array(df[x])
        y=np.array(df[y])
        plt.figure(figsize=(20, 50))
        plt.barh(x, y, height=0.2, color='lightblue')
        plt.margins(x=0, y=0)
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(save_path + save_name + ".png")
        plt.clf()
        plt.cla()
        plt.close()


def plot_bar_h(x, y,
                  ylab, xlab,
                  save_path, save_name,
                  title=" ",
                  fontsize=16):
    plt.figure(figsize=(8, 8))
    plt.barh(x, y, height=0.2, color='lightblue')
    plt.margins(x=0.1, y=0.1)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_results(x, y, data, colour, save_path, save_name,
                               xlab, ylab, title, size=0.01, xaxis_labs=None, y_lim=None,
                               fontsize=12, legend_pos="upper right", order=None):
    palette = ["plum", "cornflowerblue", "coral", "mediumaquamarine"]
    # palette = "Set2"
    # palette = "colorblind"
    # palette = "muted"
    # palette = "pastel"
    # palette = "Paired"
    sns.set_palette(palette)
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 6))

    if order:
        data['Model'] = pd.Categorical(data['Model'])
        data['Model'].cat.reorder_categories(order, inplace=True)
        data.sort_values(by="Model", inplace=True, axis=0)

    g=sns.barplot(x=x, y=y, data=data, hue=colour, palette=sns.color_palette(palette, 4))

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.legend(loc=legend_pos)

    if y_lim:
        g.set(ylim=y_lim)

    if xaxis_labs:
        g.set_xticks(range(len(data[x].unique())))
        g.set_xticklabels(xaxis_labs)

    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_long_old(x, y, data, colour, save_path, save_name, xticks, palette,
                               xlab, ylab, title, size=0.01,
                               fontsize=12, legend_loc='upper left'):
    # if len(data[colour]) > 12:
    #     palette = sns.color_palette(cc.glasbey, n_colors=len(data[colour]))
    # else:
    #     sns.set_palette("Paired")

    fig, ax = plt.subplots(figsize=(6, 6))
    # plt.figure(figsize=(10, 6))

    g = sns.pointplot(x=x, y=y, data=data, hue=colour, s=size, palette=palette, ax=ax,
                  markers='.', alpha=0.05, linewidth=0.2, legend=False)

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    g.set_xticklabels(xticks)
    # plt.legend(bbox_to_anchor=(1.01, 1.015))
    plt.legend(loc=legend_loc)
    plt.tight_layout()

    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_long(x, y, data, colour, save_path, save_name, xticks, palette,
              xlab, ylab, title, size=0.01, linewidth=4,
              fontsize=12, legend_loc='upper left'):

    fig, ax = plt.subplots(figsize=(6, 6))

    g = sns.lineplot(x=x, y=y, data=data, hue=colour, ax=ax,
                      palette=palette, linewidth=linewidth)

    # g = sns.pointplot(x=x, y=y, data=data, hue=colour, s=size, palette=palette, ax=ax,
    #                   markers='.', alpha=0.05, linewidth=0.2, legend=False)

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    g.set_xticklabels(xticks)
    # plt.legend(bbox_to_anchor=(1.01, 1.015))
    plt.legend(loc=legend_loc)
    plt.tight_layout()

    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()




def plot_perm(df, x, y, n_plot, save_path, save_name, title):
    y_ticks = np.arange(0, n_plot)
    fig, ax = plt.subplots()
    ax.barh(y_ticks, df[x])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(df[y])
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_path + save_name)
    plt.clf()
    plt.cla()
    plt.close()



def plot_perm_box(X, result, save_path, save_name, permutations,
                  xlab, ylab, title, n_plot,
                  fontsize=12, palette="Paired"):
        var_imp = pd.DataFrame(result.importances, index=X.columns)
        var_imp['Mean'] = round(var_imp.mean(axis=1), 5)
        var_imp.sort_values(by='Mean', axis=0, inplace=True, ascending=False)
        var_imp = var_imp[0:n_plot]
        var_imp.sort_values(by='Mean', axis=0, inplace=True, ascending=True)
        var_imp.drop('Mean', axis=1, inplace=True)
        var_imp = var_imp.T

        sns.set_palette(palette)
        plt.figure(figsize=(10, 6))
        plt.boxplot(var_imp, vert=False, labels=var_imp.columns)

        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)

        plt.savefig(save_path + save_name + ".png")
        plt.clf()
        plt.cla()
        plt.close()


def plot_corr(cor, save_path, save_name, title, fontsize=25, font_scale=2, figsize=(16, 14), ano=True):
    # get colourmap to center on 0:
    vcenter = 0
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=-1, vmax=1)
    cmap = sns.color_palette("RdBu_r", 100)

    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=font_scale, rc={"lines.linewidth": 5})
    sns.heatmap(cor,
                    xticklabels=cor.columns.values,
                    yticklabels=cor.columns.values,
                    cbar=True, annot=ano,
                    cmap=cmap,
                    norm=normalize)
    plt.subplots_adjust(bottom=0.5, left=0.3)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(str(save_path) + save_name, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def plot_corr_blocks(cor, save_path, save_name, title, fontsize=10, font_scale=2,
                     figsize=(20, 17), ano=False, lines=None, buffer=5,
                     label_geo=None, group_labels=None):

    vcenter = 0.5
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=0, vmax=1)
    colors = ["white", "orange", "crimson"]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Custom", colors, N=100)

    g, ax = plt.subplots()
    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=font_scale, rc={"lines.linewidth": 5})
    # Getting the lower triangle of the co-relation matrix
    matrix = np.tril(cor)

    # using the upper triangle matrix as mask
    g = sns.heatmap(cor,
                    xticklabels=cor.columns.values,
                    yticklabels=cor.columns.values,
                    cbar=True, annot=ano,
                    cmap=cmap,
                    norm=normalize,
                    cbar_kws={"shrink": 0.5},
                    mask=matrix)

    plt.title(title, fontsize=fontsize)
    g.tick_params(left=False, bottom=False)
    g.set(yticklabels=[], xticklabels=[])

    # add h/v lines in for groups
    max = len(cor.columns)+buffer
    if lines:
        g.hlines(lines, xmin=-1, xmax=max, color='black', linewidth=2)
        g.vlines(lines, ymin=-1, ymax=max, color='black', linewidth=2)

    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
                plt.text(-1, i, lab, horizontalalignment='right',
                         size='small', color='black')
                plt.text(i+0.5, -1, lab, horizontalalignment='right',
                         size='small', color='black', rotation=90)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.15, left=0.15)
    plt.savefig(str(save_path) + save_name)
    plt.clf()
    plt.cla()
    plt.close()



def plot_corr_blocks_2col(cor, save_path, save_name, title, fontsize=10, font_scale=2,
                     figsize=(20, 17), ano=False, lines=None, buffer=5,
                     label_geo=None, group_labels=None):

    # get colourmap to center on 0:
    vcenter = 0
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=-1, vmax=1)
    cmap = sns.color_palette("RdBu_r", 100)

    g, ax = plt.subplots()
    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=font_scale, rc={"lines.linewidth": 5})
    g = sns.heatmap(cor,
                    xticklabels=cor.columns.values,
                    yticklabels=cor.columns.values,
                    cbar=True, annot=ano,
                    cmap=cmap,
                    norm=normalize,
                    cbar_kws={"shrink": 0.5},
                    vmin=-1, vmax=1)

    plt.title(title, fontsize=fontsize)
    g.tick_params(left=False, bottom=False)
    g.set(yticklabels=[], xticklabels=[])

    # add h/v lines in for groups
    max = len(cor.columns)+buffer
    if lines:
        g.hlines(lines, xmin=-1, xmax=max, color='black', linewidth=2)
        g.vlines(lines, ymin=-1, ymax=max, color='black', linewidth=2)

    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
                plt.text(-1, i, lab, horizontalalignment='right',
                         size='small', color='black')
                plt.text(i+0.5, -1, lab, horizontalalignment='right',
                         size='small', color='black', rotation=90)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.15, left=0.15)
    plt.savefig(str(save_path) + save_name)
    plt.clf()
    plt.cla()
    plt.close()



def plot_corr_blocks_HH(cor, save_path, save_name, title, fontsize=10, font_scale=2,
                     figsize=(24, 17), ano=False, lines=None, buffer=5,
                     label_geo=None, group_labels=None):

    fig, ax = plt.subplots()
    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=font_scale, rc={"lines.linewidth": 5})

    # creating upper triangle
    mask1 = np.tril(np.ones_like(cor))

    # get colourmap to center on 0:
    vcenter = 0
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=-1, vmax=1)
    cmap = sns.color_palette("RdBu_r", 100)

    g = sns.heatmap(cor,
                    xticklabels=cor.columns.values,
                    yticklabels=cor.columns.values,
                    cbar=True, annot=ano,
                    cmap=cmap,
                    norm=normalize,
                    cbar_kws={"shrink": 0.5,
                              "ticks":[-1, -0.5, 0, 0.5, 1]},
                    mask=mask1)

    # creating lower triangle:
    mask2 = np.triu(cor)

    # abs:
    cor = abs(cor)

    # get colour map
    vcenter = 0.5
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=0, vmax=1)
    colors = ["white", "orange", "maroon"]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Custom", colors, N=100)

    g2 = sns.heatmap(cor,
                    xticklabels=cor.columns.values,
                    yticklabels=cor.columns.values,
                    cbar=True, annot=ano,
                    cmap=cmap,
                    norm=normalize,
                    cbar_kws={"shrink": 0.5}, mask=mask2)

    plt.title(title, fontsize=fontsize)
    g.tick_params(left=False, bottom=False)
    g.set(yticklabels=[], xticklabels=[])

    # add h/v lines in for groups
    max = len(cor.columns)+buffer
    if lines:
        g.hlines(lines, xmin=-1, xmax=max, color='black', linewidth=2)
        g.vlines(lines, ymin=-1, ymax=max, color='black', linewidth=2)

    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
                plt.text(-1, i, lab, horizontalalignment='right',
                         size='small', color='black')
                plt.text(i+0.5, -1, lab, horizontalalignment='right',
                         size='small', color='black', rotation=90)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.15, left=0.15)
    plt.savefig(str(save_path) + save_name)
    plt.clf()
    plt.cla()
    plt.close()




def plot_clust(cor, save_path, save_name, title, fontsize=14,
               font_scale=0.85, figsize=(25, 25), ano=True):
    # get colourmap to center on 0:
    vcenter = 0
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=-1, vmax=1)
    cmap = sns.color_palette("RdBu_r", 100)

    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 5})
    sns.set(font_scale=font_scale)
    sns.clustermap(cor,
                    xticklabels=cor.columns.values,
                    yticklabels=cor.columns.values,
                    annot=ano, cbar_pos=None,
                    metric='correlation',
                    cmap=cmap,
                    norm=normalize)
    plt.subplots_adjust(bottom=0.6)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(str(save_path) + save_name, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def plot_SHAP(shap_dict, col_list, plot_type, n_features,
              save_path, save_name, figsize=(6,6)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_dict, feature_names=col_list,
                      plot_type=plot_type, max_display=n_features)
    plt.tight_layout()
    plt.savefig(save_path + save_name)
    plt.clf()
    plt.cla()
    plt.close()

def plot_SHAP_over_time(x, y, data, colour, save_path, save_name, xticks, palette,
                               xlab, ylab, title, linewidth=4, legend_loc ='upper left',
                               fontsize=12, size=None, legend=True, order=None):

    data.replace(to_replace="Maths Ability Time 1", value="Prior Achievement", inplace=True)
    fig, ax = plt.subplots(figsize=(6, 6.2))
    # g = sns.pointplot(x=x, y=y, data=data, hue=colour, s=size,
    #                   palette=palette, ax=ax,
    #                   markers='.', alpha=0.05, linewidth=0.2, legend=False)
    order = ['Prior Achievement', 'School Track ', 'Class Context (T)', 'Grades',
             'Cognitive Strategies (S)', 'Family Context (S, P)', 'Demographics and SES',
              'IQ', 'Class Context (S)', 'Motivation and Emotion (S)']

    if order:
        data[colour] = pd.Categorical(data[colour])
        data[colour].cat.reorder_categories(order, inplace=True)
        data.sort_values(by=colour, inplace=True, axis=0)

    data[x] = pd.to_numeric(data[x])
    g = sns.lineplot(x=x, y=y, data=data, hue=colour, ax=ax,
                      palette=palette, linewidth=linewidth)
    plt.xticks([1, 2, 3, 4])
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.ylim(0, 22500)

    g.set_xticklabels(xticks)

    if legend==True:
        # change the line width for the legend
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles=handles[1:], labels=labels[1:], loc=legend_loc )
        for line in leg.get_lines():
            line.set_linewidth(linewidth)
    # leg.texts[0].set_text("Feature Block")
    if legend == False:
        ax.get_legend().remove()
        save_name = save_name + "_noLegend"

    plt.tight_layout()

    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_SHAP_over_time_exp(x, y, data, colour, save_path, save_name, xticks, palette,
                               xlab, ylab, title, size=0.01, legend_loc ='upper left',
                               fontsize=12, order=1):

    fig, ax = plt.subplots(figsize=(10, 10))

    x_num = data[x].apply(pd.to_numeric)
    x_num = pd.DataFrame(x_num)
    x_num.columns = ['Time_num']
    data = pd.concat([data, x_num], axis=1)
    g = sns.lmplot(x='Time_num', y=y, data=data, hue=colour, ci=None, order=order, truncate=True,
                   legend=False, palette=palette, markers='.')

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    plt.xticks([1, 2, 3, 4], labels=xticks)
    plt.legend(loc=legend_loc)
    plt.tight_layout()

    plt.savefig(save_path + save_name + "order{}".format(order) + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def get_custom_palette(df, var_name, long=False):
    unique = df[var_name].unique()
    if long == True:
        palette = dict(zip(unique, sns.color_palette("cc.glasbey", n_colors=len(unique))))
    else:
        short_palette = ["tomato", "plum", "cornflowerblue", "orange", "mediumaquamarine",
                         "mediumpurple", "sienna", "yellowgreen", "lightgrey", "gold", "gold"]
        palette = dict(zip(unique, sns.color_palette(short_palette, n_colors=11)))
    return palette


def heatmap_importance(df, save_name, save_path, index, columns, values, title,
                       xlab, ylab, xticks, gamma, fontsize=12, palette="viridis", show_n="all",
                       adjust_left=0.3, figsize=(8, 8), sort_by = "time", tick_font_size=8):
    df = pd.pivot(df, index=index, columns=columns, values=values)
    if sort_by == "time":
        df.sort_values(by=["1", "2", "3", "4"], inplace=True, ascending=False)
    elif sort_by == "overall":
        df['sum'] = df.sum(axis=1)
        df.sort_values(by="sum", inplace=True, ascending=False)
        df.drop("sum", axis=1, inplace=True)
    if show_n != "all":
        df = df.iloc[0:show_n, :]
    df.columns = xticks

    fig, ax = plt.subplots(figsize=figsize)
    # mi, ma = df.values.min(), df.values.max()
    sns.heatmap(df, cmap=palette, norm=colors.PowerNorm(gamma=gamma), yticklabels=True,
                cbar_kws={'shrink': 0.5})
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.yticks(fontsize=tick_font_size)
    plt.subplots_adjust(left=adjust_left)
    plt.savefig(save_path + save_name + ".png")
    return



def plot_scatt(x, y, save_path, save_name, xlab, ylab, data=None, fontsize=12):
    if data != None:
        x = data[x]
        y = data[y]
        x_str = x
        y_str = y
    else:
        x_str = x.name
        y_str = y.name

    # computes correlations excluding NAs
    corr = round(pd.Series(x).corr(pd.Series(y), method='pearson', min_periods=None), 2)
    print('correlation between x ({}) and y ({}) = {}'.format(x_str, y_str, corr))
    title = "Pearson r Correlation = {}".format(corr)

    plt.scatter(x, y)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_cat_scatt(x, y, save_path, save_name, xticks,
                   name=None, binary=True,
                   xlab='', ylab='',
                   data=None, title='',
                   leg_title="School Track",
                   add_means=True,
                   fontsize=12):
    plt.figure(figsize=(6, 6.5))
    g = sns.swarmplot(x=round(data[x],0), y=data[y].astype("category"), data=data, hue=y,
                  orient="h", size=3)
    plt.title(title, fontsize=fontsize)
    if binary == True:
        plt.legend(title=leg_title, labels=['Not', name])
    else:
        plt.legend(title=leg_title, labels=['Gymnasium', "Realshule", "Hauptshule"])

        if add_means == True:
            plt.axvline(x=np.mean(data[data[y]==1][x]),
                        ls="-", color='royalblue', alpha=0.9)
            plt.axvline(x=np.mean(data[data[y]==2][x]),
                        ls="-", color='darkorange', alpha=0.9)
            plt.axvline(x=np.mean(data[data[y]==3][x]),
                        ls="-", color='green', alpha=0.9)


    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    # plt.xticks(np.arange(round(min(data[x], 0)), round(max(data[x]), 0) + 1, 50))
    plt.xticks(xticks)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_SHAP_interaction(shap_interaction, save_path, save_name, col_list, plot_cols = 'all',
                          xlab="", ylab="", title="", fontsize=12):
    # Get absolute mean of matrices
    mean_shap = np.abs(shap_interaction).mean(0)
    df = pd.DataFrame(mean_shap, index=col_list, columns=col_list)

    # times off-diagonal by 2
    df.where(df.values == np.diagonal(df), df.values * 2, inplace=True)

    if plot_cols != 'all':
        # plot only highest values or cols in list
        df_reduced = df[plot_cols]
        df_reduced = df_reduced.loc[plot_cols]

    # plot
    plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    sns.set(font_scale=1)
    sns.heatmap(df_reduced, cmap='coolwarm', annot=True, fmt='.3g', cbar=False)
    plt.yticks(rotation=0)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_SHAP_summary_interaction(shap_interaction, X, save_path, save_name, col_list="",
                                  xlab="", ylab="", title="", fontsize=10):
    X = pd.DataFrame(X, columns=col_list)
    shap.summary_plot(shap_interaction, X)
    plt.yticks(rotation=0)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def plot_SHAP_interact_dependency(shap_interaction, var1, var2, X, save_path, save_name,
                                  xlab="", ylab="", title="", fontsize=10):
    shap.dependence_plot(
        (var1, var2),
        shap_interaction, X,
        display_features=X)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return



def annotate_yranges(groups, ax=None):
    """
    Annotate a group of consecutive yticklabels with a group name.

    Arguments:
    ----------
    groups : dict
        Mapping from group label to an ordered list of group members.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """
    if ax is None:
        ax = plt.gca()

    label2obj = {ticklabel.get_text() : ticklabel for ticklabel in ax.get_yticklabels()}

    for ii, (group, members) in enumerate(groups.items()):
        first = members[0]
        last = members[-1]

        bbox0 = _get_text_object_bbox(label2obj[first], ax)
        bbox1 = _get_text_object_bbox(label2obj[last], ax)

        set_yrange_label(group, bbox0.y0 + bbox0.height/2,
                         bbox1.y0 + bbox1.height/2,
                         min(bbox0.x0, bbox1.x0),
                         -2,
                         ax=ax)


def set_yrange_label(label, ymin, ymax, x, dx=-0.5, ax=None, *args, **kwargs):
    """
    Annotate a y-range.

    Arguments:
    ----------
    label : string
        The label.
    ymin, ymax : float, float
        The y-range in data coordinates.
    x : float
        The x position of the annotation arrow endpoints in data coordinates.
    dx : float (default -0.5)
        The offset from x at which the label is placed.
    ax : matplotlib.axes object (default None)
        The axis instance to annotate.
    """

    if not ax:
        ax = plt.gca()

    dy = ymax - ymin
    props = dict(connectionstyle='angle, angleA=90, angleB=180, rad=0',
                 arrowstyle='-',
                 shrinkA=10,
                 shrinkB=10,
                 lw=1)
    ax.annotate(label,
                xy=(x, ymin),
                xytext=(x + dx, ymin + dy/2),
                annotation_clip=False,
                arrowprops=props,
                *args, **kwargs,
    )
    ax.annotate(label,
                xy=(x, ymax),
                xytext=(x + dx, ymin + dy/2),
                annotation_clip=False,
                arrowprops=props,
                *args, **kwargs,
    )


def _get_text_object_bbox(text_obj, ax):
    # https://stackoverflow.com/a/35419796/2912349
    transform = ax.transData.inverted()
    # the figure needs to have been drawn once, otherwise there is no renderer?
    plt.ion(); plt.show(); plt.pause(0.001)
    bb = text_obj.get_window_extent(renderer = ax.get_figure().canvas.renderer)
    # handle canvas resizing
    return TransformedBbox(bb, transform)



def plot_shap_scatt(data, x, y, save_path, save_name, xlab, ylab,
                    hue=None, fontsize=12, title ="",
                    lines=None, buffer=10,
                    label_geo=None, group_labels=None
                    ):
    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 3))

    vcenter = 0
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter,
                                     vmin=data[y].min(), vmax=data[y].max())
    cmap = sns.color_palette("RdYlBu", 5)
    # pal = {"3": "yellow", "5": "darkblue", "4": "lightblue",
    #        "2": "orange", "1": "red"}
    # pal = ["blue", "lightblue", "yellow", "orange", "red"]
    # cmap = ListedColormap(pal)

    g = sns.stripplot(data=data, x=x, y=y, hue=hue, order=None, hue_order=None,
                      jitter=False, dodge=False, orient=None, color=None, palette=cmap,
                      size=4, edgecolor='gray', linewidth=0, alpha=.8)
                      # hue_norm=normalize, native_scale=False, formatter=None,
                      # legend='auto', ax=None)

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    g.tick_params(left=False, bottom=False)
    g.set(xticklabels=[])

    # add h/v lines in for groups
    ymax = data['SHAP_value'].max() + buffer
    ymin = data['SHAP_value'].min() - buffer

    if lines:
        g.vlines(lines, ymin=ymin, ymax=ymax, color='black', linewidth=1)

    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
            plt.text(i, ymax +buffer, lab, horizontalalignment='center',
                     verticalalignment = "bottom",
                     size='small', color='black', rotation=45)

    plt.legend(loc="upper right", title="Variable value",
               # labels=['High', ' ','Medium',' ', 'Low'])
               labels={"1":'High', "2":' ', "3":'Medium', "4":' ', "5":'Low'})
    # todo: create continous colorbar, +non-lin y axis
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_shap_scatt_looped(data, x, y, save_path, save_name, xlab, ylab,
                    hue=None, fontsize=11, title ="",
                    lines=None, buffer=5, non_lin=False,
                    label_geo=None, group_labels=None, ymax = 75
                    ):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

    #original colours: ["red", "violet", "blue"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "violet", "firebrick"], N=7)
    ymin = data['SHAP_value'].min() - buffer

    data = data[data.data_value < 5]
    axes_list = [0, 1, 2, 3]
    for ax in axes_list:
        time = ax+1
        print("plotting {}".format(time))
        df = data[data['Time']==time]
        print(df.data_value.min())
        print(df.data_value.max())
        im = axes[ax].scatter(data=df, x=x, y=y,
                    c=hue, edgecolor='none', alpha=1, s=15, cmap=cmap)

        axes[ax].set_ylabel(ylab, fontsize=7)
        axes[ax].tick_params(left=False, bottom=False)
        axes[ax].set(xticklabels=[])
        axes[ax].margins(x=0.01)
        for spine in axes[ax].spines.values():
            spine.set_edgecolor('lightgrey')
        if non_lin == True:
            buffer = 1000000000
            axes[ax].set_yscale('symlog', linthresh=0.001)
            axes[ax].set_yticks([-1000, 0, 10000])
            axes[ax].set_yticklabels(['_', 'o', '+'])
        else:
            axes[ax].set_ylim(-75, ymax)
        axes[ax].legend([], [], frameon=False)
        if lines:
            axes[ax].vlines(lines, ymin=ymin, ymax=ymax, color='black', linewidth=1)

    axes[3].set_xlabel(xlab, fontsize=fontsize)
    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
            axes[0].text(i, ymax + buffer, lab, horizontalalignment='center',
                     verticalalignment = "bottom",
                     size='small', color='black', rotation=60)

    fig.subplots_adjust(right=0.97)
    cbar = fig.colorbar(im, ax=axes[:], shrink=0.3, location='right')
    cbar.set_ticks([-2.8, 4])
    cbar.ax.set_yticklabels(['low', 'high'], fontsize=8)
    cbar.ax.set_title('Variable value', fontsize=10)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()

def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

def plot_shap_swarm_looped(data, x, y, save_path, save_name, xlab, ylab,
                    hue=None, fontsize=11, title ="",
                    lines=None, buffer=5, non_lin=False,
                    label_geo=None, group_labels=None, ymax = 75
                    ):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

    #original colours: ["red", "violet", "blue"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "violet", "firebrick"], N=7)
    ymin = data['SHAP_value'].min() - buffer

    data = data[data.data_value < 5]
    axes_list = [0, 1, 2, 3]
    for ax in axes_list:
        time = ax+1
        print("plotting {}".format(time))
        df = data[data['Time']==time]
        print(df.data_value.min())
        print(df.data_value.max())
        x_points = simple_beeswarm(df[y])
        im = axes[ax].plot(x_points, df[y], "o" )

                    # c=hue, edgecolor='none', alpha=1, s=15, cmap=cmap)

        axes[ax].set_ylabel(ylab, fontsize=7)
        axes[ax].tick_params(left=False, bottom=False)
        axes[ax].set(xticklabels=[])
        axes[ax].margins(x=0.01)
        for spine in axes[ax].spines.values():
            spine.set_edgecolor('lightgrey')
        if non_lin == True:
            buffer = 1000000000
            axes[ax].set_yscale('symlog', linthresh=0.001)
            axes[ax].set_yticks([-1000, 0, 10000])
            axes[ax].set_yticklabels(['_', 'o', '+'])
        else:
            axes[ax].set_ylim(-75, ymax)
        axes[ax].legend([], [], frameon=False)
        if lines:
            axes[ax].vlines(lines, ymin=ymin, ymax=ymax, color='black', linewidth=1)

    axes[3].set_xlabel(xlab, fontsize=fontsize)
    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
            axes[0].text(i, ymax + buffer, lab, horizontalalignment='center',
                     verticalalignment = "bottom",
                     size='small', color='black', rotation=60)

    fig.subplots_adjust(right=0.97)
    # cbar = fig.colorbar(im, ax=axes[:], shrink=0.3, location='right')
    # cbar.set_ticks([-2.8, 4])
    # cbar.ax.set_yticklabels(['low', 'high'], fontsize=8)
    # cbar.ax.set_title('Variable value', fontsize=10)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_shap_sns_swarm_looped(data, x, y, save_path, save_name, xlab, ylab,
                           hue=None, fontsize=11, title="",
                           lines=None, buffer=5, non_lin=False,
                           label_geo=None, group_labels=None, ymax=75
                           ):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

    # original colours: ["red", "violet", "blue"]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "violet", "firebrick"], N=7)
    ymin = data['SHAP_value'].min() - buffer
    cmap = sns.color_palette("RdYlBu", 5)

    data = data[data.data_value < 5]
    axes_list = [0, 1, 2, 3]
    for ax in axes_list:
        time = ax + 1
        print("plotting {}".format(time))
        df = data[data['Time'] == time]
        print(df.data_value.min())
        print(df.data_value.max())
        im = sns.swarmplot(ax=axes[ax], data=df, x=x, y=y, hue=hue, order=None, hue_order=None,
                  dodge=False, orient=None, color=None, palette=cmap,
                  size=2, edgecolor='gray', linewidth=0, alpha=1)

        axes[ax].set_ylabel(ylab, fontsize=7)
        axes[ax].set_xlabel(" ")
        axes[ax].tick_params(left=False, bottom=False)
        axes[ax].set(xticklabels=[])
        axes[ax].margins(x=0.01)
        for spine in axes[ax].spines.values():
            spine.set_edgecolor('lightgrey')
        if non_lin == True:
            buffer = 1000000000
            axes[ax].set_yscale('symlog', linthresh=0.001)
            axes[ax].set_yticks([-1000, 0, 10000])
            axes[ax].set_yticklabels(['_', 'o', '+'])
        else:
            axes[ax].set_ylim(-75, ymax)
        axes[ax].legend([], [], frameon=False)
        if lines:
            axes[ax].vlines(lines, ymin=ymin, ymax=ymax, color='black', linewidth=1)
    #     todo: ensure lines overlay

    axes[3].set_xlabel(xlab, fontsize=fontsize)
    if (label_geo) and (group_labels):
        for i, lab in zip(label_geo, group_labels):
            axes[0].text(i, ymax + buffer, lab, horizontalalignment='center',
                         verticalalignment="bottom",
                         size='small', color='black', rotation=60)

    fig.subplots_adjust(right=0.97)

    # norm = plt.Normalize(tips['size'].min(), tips['size'].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r")
    sm.set_array([])

    # add a colorbar
    cbar = im.figure.colorbar(sm, ax=axes[:], shrink=0.3, location='right', boundaries=[-2,-1,-0.5,0.5,1,2])
    # cbar.set_ticks([-2, 2])
    cbar.ax.set_yticklabels(['low', ' ', ' ', ' ', ' ', 'high'], fontsize=8)
    cbar.ax.set_title('Variable value \n', fontsize=10)

    # bounds = [-1, 2, 5, 7, 12, 15]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    # cbar = plt.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    #                                spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()

# todo: try smaller dots and better groupings of colours (so all 1s and -1s are in same group