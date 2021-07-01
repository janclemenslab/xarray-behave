import xarray_behave as xb
import logging

logging.getLogger().setLevel(logging.INFO)

recs = {'rpi9-20210409_093149': {},
        'localhost-20210617_113024': {},
        'localhost-20181120_144618': {},
        'localhost-20210624_104612': {},
        'localhost-20210628_145223': {'dat_path': 'dat', 'res_path': 'res', 'root': '/Volumes/ukme04/#Data/flyball/'},
        'localhost-20210629_171532': {'dat_path': 'dat', 'res_path': 'res', 'root': '/Volumes/ukme04/#Data/flyball/'},
        'Dmel_male': {'dat_path': 'dat', 'res_path': 'dat', 'root': '.', 'filepath_daq': 'dat/Dmel_male.wav', 'filepath_annotations': 'dat/Dmel_male_annotations.csv'},
        'Dmel_male2': {'dat_path': 'dat', 'res_path': 'dat', 'root': '.', 'filepath_daq': 'dat/Dmel_male.npz', 'filepath_annotations': 'dat/Dmel_male_annotations.csv'}}

for datename, kwargs in recs.items():
    # print(datename)
    ds = xb.assemble(datename, **kwargs)
    # print(ds)

def assemble0():
    datenames = list(recs.keys())
    ii = 0
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

def assemble1():
    datenames = list(recs.keys())
    ii = 1
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

def assemble2():
    datenames = list(recs.keys())
    ii = 2
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

def assemble3():
    datenames = list(recs.keys())
    ii = 3
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

def assemble4():
    datenames = list(recs.keys())
    ii = 4
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)


def assemble5():
    datenames = list(recs.keys())
    ii = 5
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

def assemble6():
    datenames = list(recs.keys())
    ii = 6
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

def assemble3():
    datenames = list(recs.keys())
    ii = 7
    datename = datenames[ii]
    kwargs = recs[datename]
    ds = xb.assemble(datename, **kwargs)

# def assemble4():
#     datenames = list(recs.keys())
#     ii = 8
#     datename = datenames[ii]
#     kwargs = recs[datename]
#     ds = xb.assemble(datename, **kwargs)


# root = ''
# datenames = ['rpi9-20210409_093149', 'localhost-20210617_113024', 'localhost-20181120_144618', 'localhost-20210624_104612']
# # datenames = ['localhost-20210624_104612']
# # datenames = ['rpi9-20210409_093149']
# datenames = ['localhost-20210624_111045']
# # datenames = ['localhost-20210628_145223']
# # datenames = ['localhost-20210629_171532']
# # for datename in datenames:
# #     print(datename)
# #     print(f'   assembling data')
# #     ds = xb.assemble(datename, dat_path='dat', res_path='res', root='/Volumes/ukme04/#Data/flyball/')
# #     print(ds)

# datenames = ['Dmel_male']
# # for datename in datenames:
#     print(datename)
#     print(f'   assembling data')
#     ds = xb.assemble(datename, dat_path='dat', res_path='dat', root='', filepath_daq='dat/Dmel_male.npz', filepath_annotations='dat/Dmel_male_annotations.csv')
#     ds = xb.assemble(datename, dat_path='dat', res_path='dat', root='', filepath_daq='dat/Dmel_male.wav', filepath_annotations='dat/Dmel_male_annotations.csv')
#     print(ds)
