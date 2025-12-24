#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhou Ze'
__version__ = '2.0'


import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from lmfit import models
from scipy import stats


def peak_decov(x, y, peaks_cen, max_scale=None):
	'''
	x: array of sizes.
	y: array of numbers or frequencies.
	peaks_cen: array of roughtly locations.
	Without scale constrains.'''
	x, y = np.array(x), np.array(y)/sum(y)*100
	model_ar, model, params = [], None, None
	for peak in peaks_cen:
		mod = models.LorentzianModel(prefix='peak{}_'.format(peak))
		mod.set_param_hint('center', value=peak,)
		mod.set_param_hint('amplitude', min=0)
		mod.set_param_hint('sigma', max=max_scale)
		par = mod.make_params()
		model_ar.append(mod)
		if model is None:
			model = mod
			params = par
		else:
			model = model + mod
			params.update(par)
	result = model.fit(y, params, x=x)

	result_par = {}
	for peak in peaks_cen:
		loc = result.values['peak{}_center'.format(peak)]  # loc
		scale = result.values['peak{}_sigma'.format(peak)]  # width
		amp = result.values['peak{}_amplitude'.format(peak)]  # area
		result_par.setdefault(loc, (scale, amp))

	final_res = []
	for loc, (scale, amp) in sorted(result_par.items()):
		final_res.append((loc, scale, amp))
	return final_res


def peak_decov_constrain(fig_x, fig_y, peaks_cen, max_scale=None):
	'''With scale constrains.'''
	fig_x, fig_y = np.array(fig_x), np.array(fig_y)/sum(fig_y)*100
	model_ar, model, params = [], None, None
	for peak in peaks_cen:
		mod = models.LorentzianModel(prefix='peak{}_'.format(peak))  # GaussianModel StudentsTModel
		mod.set_param_hint('center', value=peak)
		mod.set_param_hint('amplitude', min=0)

		if 60 in peaks_cen and 70 in peaks_cen and peak == 70:
			mod.set_param_hint('sigma', expr='peak60_sigma')
		elif 80 in peaks_cen and 90 in peaks_cen and peak == 90:
			mod.set_param_hint('sigma', expr='peak80_sigma')
		elif 101 in peaks_cen and 111 in peaks_cen and peak == 111:
			mod.set_param_hint('sigma', expr='peak101_sigma')
		elif 121 in peaks_cen and 131 in peaks_cen and peak == 131:
			mod.set_param_hint('sigma', expr='peak121_sigma')
		elif 141 in peaks_cen and 151 in peaks_cen and peak == 151:
			mod.set_param_hint('sigma', expr='peak141_sigma')
		else:
			mod.set_param_hint('sigma', max=max_scale)

		par = mod.make_params()
		model_ar.append(mod)
		if model is None:
			model = mod
			params = par
		else:
			model = model + mod
			params.update(par)
	result = model.fit(fig_y, params, x=fig_x)
	print(result.fit_report(), f'\nAIC {result.aic}', f'\nBIC {result.bic}')

	result_par = {}
	for peak in peaks_cen:
		loc = result.values['peak{}_center'.format(peak)]  # loc
		scale = result.values['peak{}_sigma'.format(peak)]  # width
		amp = result.values['peak{}_amplitude'.format(peak)]  # area
		result_par.setdefault(loc, (scale, amp))

	final_res = []
	for loc, (scale, amp) in sorted(result_par.items()):
		final_res.append((loc, scale, amp))
	return final_res


def peak_decov_l2_regularization(fig_x, fig_y, peaks_cen, min_scale=2, max_scale=8,lam=0.00002):
	from lmfit import Minimizer, Parameters, report_fit
	from lmfit.lineshapes import gaussian, lorentzian
	from lmfit import models

	fig_x, fig_y = np.array(fig_x), np.array(fig_y)/sum(fig_y)*100
	def residual(pars, x, data):
		model = None
		for idx in range(len(peaks_cen)):
			peak = peaks_cen[idx]
			mod = lorentzian(
				x,
				pars[f'peak{peak}_amplitude'],
				pars[f'peak{peak}_center'],
				pars[f'peak{peak}_sigma'])
			if model is None:
				model = mod
			else:
				model = model + mod

		sigmas = []
		for peak in peaks_cen:
			name = f'peak{peak}_sigma'
			if name in pars and not pars[name].expr:  # skip those with expr (linked)
				sigmas.append(pars[name].value)
		sigmas = np.array(sigmas)

		return model-data+ lam*np.sum(sigmas)**2

	pfit = Parameters()
	for idx in range(len(peaks_cen)):
		peak = peaks_cen[idx]

		pfit.add(name=f'peak{peak}_amplitude', min=0.01)
		pfit.add(name=f'peak{peak}_center', value=peaks_cen[idx], min=peaks_cen[idx]-3, max=peaks_cen[idx]+3)
		#pfit.add(name=f'peak{peak}_center', value=peaks_cen[idx])

		if 60 in peaks_cen and 70 in peaks_cen and peak == 70:
			mod.set_param_hint('sigma', expr='peak60_sigma')
		elif 90 in peaks_cen and 80 in peaks_cen and peak == 90:
			mod.set_param_hint('sigma', expr='peak80_sigma')
		elif 111 in peaks_cen and 101 in peaks_cen and peak == 111:
			mod.set_param_hint('sigma', expr='peak101_sigma')
		elif 131 in peaks_cen and 121 in peaks_cen and peak == 131:
			pfit.add(name='peak{}_sigma'.format(peak), expr='peak121_sigma')
		elif 151 in peaks_cen and 141 in peaks_cen and peak == 151:
			pfit.add(name='peak{}_sigma'.format(peak), expr='peak141_sigma')
		elif peak >= 177:
			pfit.add(name='peak{}_sigma'.format(peak), expr='peak167_sigma')
		else:
			pfit.add(name='peak{}_sigma'.format(peak),  min=min_scale, max=max_scale)

	mini = Minimizer(residual, pfit, fcn_args=(fig_x, fig_y))
	result = mini.least_squares()
	pars = result.params

	final_res = []
	for idx in range(len(peaks_cen)):
		peak_cen = '{}'.format(peaks_cen[idx]).replace('.', '_')
		loc, scale, amp = pars['peak{}_center'.format(peak_cen)].value,\
			pars['peak{}_sigma'.format(peak_cen)].value,\
			pars['peak{}_amplitude'.format(peak_cen)].value
		final_res.append((loc, scale, amp))
	#delta_sig = pars['delta_sig'].value
	return final_res


def color_iter(num, color_family, min_frac, max_frac):
	color_family = eval("cm.{}".format(color_family))
	return iter(color_family(np.linspace(min_frac, max_frac, num)))


def plot_peaks(x, y, pars, fig_file='size_decov.pdf', title='', ylim_max=None):
	x, y = np.array(x), np.array(y)/sum(y)*100
	model_ar, model, params = [], None, None
	x_high = np.linspace(min(x), max(x), num=1000)
	fig, axs = plt.subplots(
		2, 1,
		sharex=True,
		gridspec_kw={'height_ratios': [20, 1]}, 
		figsize=(6, 6))

	axs[0].plot(
		x,
		y,
		color='silver',
		linestyle='-',
		label='Size profile',
		linewidth=4,
		alpha=1)

	best_fit = np.array([0. for x in x])
	colors = color_iter(len(pars), 'spring', 0., 0.85)
	cols = list(colors)

	for par, col in zip(pars, cols):
		loc, scale, amp = par[:3]
		dist = stats.cauchy(loc=loc, scale=scale)
		fit = dist.pdf(x) * amp
		best_fit += fit
		y_high = dist.pdf(x_high)*amp 
		axs[0].plot(
			x_high,
			y_high,
			color=col,
			linestyle='-',
			linewidth=1.5)

	axs[0].plot(
		x,
		best_fit,
		color='k',
		linestyle='--',
		label='Best fit',
		linewidth=1)

	if ylim_max is None:
		axs[0].set_yticks([_ for _ in range(int(round(max(y)))+1)])
		axs[0].set_yticklabels([f'{_}' for _ in range(int(round(max(y)))+1)])
	else:
		axs[0].set_yticks([_ for _ in range(ylim_max+1)])
		axs[0].set_yticklabels([f'{_}' for _ in range(ylim_max+1)])
	axs[0].tick_params(axis='y', labelsize=15)
	axs[0].set_ylim(ymin=0)
	axs[0].set_ylabel('Frequency (%)', fontsize=20)
	axs[0].tick_params(axis="y", labelsize=15)

	for par, col in zip(pars, cols):
		loc, scale, amp = par[:3]
		dist = stats.cauchy(loc=loc, scale=scale)

		axs[1].scatter(
			[loc],
			[0],
			color=col,
			s=15 if 158 < loc < 161 else 10,
			marker='^' if 158 < loc < 161 else 'o',
			facecolor='none' if 158 < loc < 161 else col,
			)  # center

		axs[1].hlines(
			-0.5,
			loc-scale/2,
			loc+scale/2,
			color=col,
			lw=1)
	axs[1].set_ylim(-0.8, 0.5)

	axs[1].set_yticks([])
	axs[1].set_xlabel('Size (bp)', fontsize=20)
	axs[1].tick_params(axis="x", labelsize=15)

	fig.suptitle(title, fontsize=20)
	plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.15, hspace=0)
	plt.savefig(fig_file)


if __name__ == '__main__':
	x, y = [], []
	with open('./example_size.txt') as f:
		for line in f:
			size, num = line.rstrip().split()
			size, num = int(size), float(num)
			x.append(size)
			y.append(num)

	# ordinary size deconvolution
	print('# Start cfDNA size deconvolution')
	result_pars = peak_decov(
		x,
		y,
		[60,70,80,90,101,111,121,131,141,151,159,167,177,188,199],
		max_scale=8,
		)
	print('# Finished')

	print('#Without constrain')
	print('#Location\tScale (bp)\tAmplitude (%)')
	for loc, scl, amp in (result_pars):
		print(f'{loc:.1f}\t{scl:.2f}\t{amp:.2f}')
	plot_peaks(x, y, result_pars, 'size_decov_ordinary.png')


	# size deconvolution with constrains
	print('# Start cfDNA size deconvolution with constrains')
	result_pars = peak_decov_constrain(
		x,
		y,
		[60,70,80,90,101,111,121,131,141,151,159,167,177,188,199,210],
		max_scale=8,
		)
	print('# Finished')

	print('#With scale constrains')
	print('#Location\tScale (bp)\tAmplitude (%)')
	for loc, scl, amp in (result_pars):
		print(f'{loc:.1f}\t{scl:.2f}\t{amp:.2f}')
	plot_peaks(x, y, result_pars, 'size_decov_constrains.png')

	######  size deconvolution with constrains and L2 regularization   ######
	x, y = [], []
	with open('./example_size_sWGS.txt') as f:
		for line in f:
			size, num = line.rstrip().split()
			size, num = int(size), float(num)
			x.append(size)
			y.append(num)

	# size deconvolution with constrains and L2 regularization
	print('# Start cfDNA size deconvolution with constrains and L2 regularization')
	result_pars = peak_decov_l2_regularization(
		x,
		y,
		peaks_cen = [121,131,141,151,159,167,177,188,199],
		)
	print('# Finished')

	print('#With scale constrains')
	print('#Location\tScale (bp)\tAmplitude (%)')
	for loc, scl, amp in (result_pars):
		print(f'{loc:.1f}\t{scl:.2f}\t{amp:.2f}')
	plot_peaks(x, y, result_pars, 'size_decov_L2_regularization.png')
