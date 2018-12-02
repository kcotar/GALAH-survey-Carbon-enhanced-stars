import matplotlib as mpl
mpl.rcParams['font.size'] = 15

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un
from astropy.table import Table, vstack, join, unique
from glob import glob
from helper_functions import get_spectra_dr52, spectra_normalize
from astropy.modeling import models, fitting

MOVE_PLOTS = False
CANNON_PLOTS = True
TSNE_PLOTS = True
APASS_PLOTS = False
# futher investigation of spectra
Li_LINES = False
H_LINES = False
REPEATS_ALL = True
FINAL_TABLE = False  # create data to be published

galah_data_dir = '/data4/cotar/'
dr53_dir = galah_data_dir+'dr5.3/'
tsne_data_dir = 'Data/'
galah_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_20180327.fits')
cannon_param = Table.read(galah_data_dir + 'sobject_iraf_iDR2_180325_cannon.fits')

results_paths = 'results_swan_line.csv'
results_folders = galah_data_dir+'Swan_band_strength_all-spectra/'
results_swan_all = Table.read(results_folders+results_paths)

# join with galah reduction data
results = join(results_swan_all, galah_data['sobject_id', 'red_flag', 'snr_c1_guess', 'snr_c1_iraf', 'ra', 'dec', 'galah_id'], keys='sobject_id', join_type='left')
# remove known reduction problems
results = results[np.logical_and(np.logical_and(np.bitwise_and(results['red_flag'], 64) == 0,
                                                np.bitwise_and(results['red_flag'], 1) == 0),
                                 results['snr_c1_guess'] >= 15)]

print 'Number of object before filtering:', len(results)
# thresholds to filter out bad fits, bad data or else something peculiar
idx_useful = np.logical_and(results['flag'] >= 0,
                            np.isfinite(results['swan_fit_integ']))
idx_wvl_ok = np.logical_not(np.logical_or(results['wvl'] >= 4738.,
                                          results['wvl'] <= 4736.))
idx_param_ok = np.logical_and(results['amp'] <= 1.,
                              results['sig'] <= 1.)
idx_param_ok2 = np.logical_and(results['swan_integ'] >= 0.1,
                               results['amp_lin'] > -0.003)
idx_use = np.logical_and(np.logical_and(idx_useful,
                                        idx_wvl_ok),
                         np.logical_and(idx_param_ok,
                                        idx_param_ok2))
results_good = results[idx_use]
print 'Number of object after filtering:', len(results_good)

# sort by strength of swan band
idx_sorted = np.flipud(np.argsort(results_good['swan_fit_integ', 'swan_integ'], order=['swan_fit_integ', 'swan_integ']))
res_final = results_good[idx_sorted]

num = 0
selected_objects = res_final[:400]
integ_sel = np.array(selected_objects['sobject_id'])

if MOVE_PLOTS:
    for object in selected_objects:  # already sorted from strongest to weakest swan fit/enhancement
        s_id = object['sobject_id']
        ra = object['ra']
        dec = object['dec']
        print num, s_id
        id_str = str(s_id)

        cannon_obj_data = cannon_param[cannon_param['sobject_id'] == s_id]
        feh = np.nan
        feh_suf = ''
        if len(cannon_obj_data) == 1:
            feh = cannon_obj_data['Fe_H_cannon'].data[0]
            if cannon_obj_data['flag_cannon'] == 0:
                if feh < -1.:
                    print cannon_obj_data['C_abund_cannon', 'flag_C_abund_cannon']
            else:
                feh_suf = 'flag'

        input_file = results_folders + str(np.abs(np.int32(s_id/10e10))) + '/' + id_str + '.png'
        if os.path.isfile(input_file):
            os.system('cp '+input_file + ' ' + results_folders + 'strong_supervised/{:04.0f}_'.format(num)+id_str+'_ra{:f}_dec{:f}'.format(ra,dec)+'_fehcannon_{:.2f}'.format(feh)+feh_suf+'.png')
        num += 1

# manual selection of objects - useful for results of tSNE analysis
tsne_input_files = ['swan_4720_4890_perp_20_theta_0.5_allok.fits']
tsne_sel = np.sort([
131118002901049,131118002901171,131119001201099,131119001201207,131119001201388,131120002501220,131120003501182,131121001901190,131122000301278,131123002501276,131123003001004,131123003501355,131123004101252,131216001101344,131216001601074,131216001601251,131217003301016,131218001901074,131218002401010,131218002401081,140111002101074,140114003701268,140115003101086,140115003101263,140116003201221,140116003201263,140118002001247,140209002201027,140209002201160,140304001801068,140304001801230,140305003201109,140305003201323,140308001401387,140309002101110,140309002601387,140309003101079,140309004701227,140310003801310,140311005601389,140311006601092,140311008701227,140312001701041,140312001701314,140312002701132,140312003001193,140312003501224,140313005201373,140314003201132,140314003601145,140314005201166,140314005201392,140315002501166,140315002501392,140316004201166,140316004201392,140316004601328,140409003001203,140412000201043,140412000201275,140413003201317,140413004901086,140414002001064,140415002401009,140608001401345,140608003101038,140608003101186,140608003101390,140608004301117,140608004301343,140609001101320,140609001101329,140609002101117,140609004901121,140610004401294,140610005001087,140610005701280,140611001601029,140611002501170,140611003001132,140611003501111,140707000101273,140707001101257,140707002101310,140707002101324,140707002601381,140707003101212,140707004101242,140708000601210,140708001201302,140708004101286,140708005301244,140708007101059,140709003801388,140710000101209,140710004601356,140711001901199,140711004401105,140713004001119,140805000901056,140805003601086,140805004801299,140806000601127,140806001701332,140806003501203,140807001101182,140807005601142,140808002701230,140808003701280,140809001601072,140809003101097,140809004901042,140810004201162,140810004701360,140810004701398,140811002101333,140811004501178,140811004501348,140811005001136,140812002101179,140812002601084,140812003201244,140812003201325,140812003201330,140812003801061,140813002201206,140814004801043,140814006601358,140823002701208,140824006301237,141101001801149,141101003501019,141102002701040,141103002601328,141104002801163,141104003301217,141104004801060,141104004801265,141231003001118,141231003001270,141231003001284,141231004001295,150101002901337,150101002901352,150102003701215,150105002801355,150106002701335,150107002201154,150107002701262,150107004701257,150108001501170,150108002801394,150109001001334,150112002501261,150204002401208,150204003701172,150205002601283,150205003401373,150205005001013,150206003601093,150207002101170,150207002101238,150207002601205,150207002601215,150208002701341,150208004701001,150208004701224,150209002201255,150209003301041,150210002201181,150210002701339,150210003201020,150210003201331,150210003701114,150210004201187,150211002201389,150211002701030,150211002701254,150211003201275,150211004701167,150211004701199,150211004701318,150330002201225,150401004101391,150405001501038,150405002001027,150405003401329,150407000601038,150408003501108,150409003601220,150409007601048,150409007601105,150412003601009,150413004101358,150413004601213,150413005601059,150426000601119,150427001301377,150427001801225,150427003301196,150428000101379,150428002101180,150428002101390,150428003101180,150428003101390,150429001101138,150429003101199,150430002301075,150430003301223,150504001901141,150504003001051,150601004801163,150601004801253,150601004801270,150602002701237,150603003301004,150603003801298,150604003401173,150605002701158,150606001901014,150606002401144,150606002401338,150606005401298,150606005901362,150607002101191,150607002601084,150607003601074,150607004101171,150607004101295,150607005101003,150607005101105,150703001601328,150703002601110,150703002601386,150704002301029,150705001901015,150705001901211,150705002901294,150705002901318,150705003401097,150705003901379,150706003901297,150706003901319,150706004401097,150706004401287,150706004401318,150706005401017,150706005901121,150826002101025,150827002901266,150827003401187,150827004001099,150828004701064,150828004701079,150828005701059,150829002601346,150830005601133,150830006101358,150831002001076,150831002001336,150831002501005,150831002501128,150831002501317,150831002501367,150831003501070,150831004001344,150901003001056,150903002401282,151008001601347,151008002601048,151008002601063,151109002101111,151109003601083,151111002601109,151219003101384,151220001601285,151220001601340,151223001301153,151225001601088,151225001601161,151225001601237,151225003301082,151225004301256,151227004701085,151229004001034,151229004001296,151229004501227,151229005501089,151230001601335,151230002201388,151230003301057,151231002601360,151231004901318,160106003101336,160108002001086,160108002001331,160109002001139,160109002701064,160110002601171,160110003101348,160111002101020,160111002101293,160111002601144,160113001601001,160113001601377,160113001601380,160113002901317,160123002101051,160123002601145,160125001601331,160125002401331,160125003001202,160125003501121,160126002601185,160129004201149,160129004201213,160129004701263,160130003601199,160325003701153,160326000101106,160326000601352,160326000601391,160326001101136,160326002101018,160327003101268,160327005601318,160328000701094,160328001101346,160328003201264,160328003601039,160328003601055,160328003601059,160330001601197,160330102301149,160331001701089,160331004801237,160401001601345,160401002601202,160401004401056,160401004401197,160401005401225,160402003601184,160402004601084,160402005101118,160402006601170,160415002101379,160415004101387,160417000101257,160417002201071,160418002101186,160418003601353,160418005601274,160418005601351,160419002101013,160419002101208,160419003101016,160419004601088,160419005101143,160419005701087,160420004801089,160420005301116,160420006301105,160420006901146,160421002101276,160421005101218,160421005601110,160423004401196,160424002601195,160424002601344,160424005701052,160425002501058,160426006101025,160514001801097,160514002301082,160514002801085,160519004101055,160519004601264,160519004601398,160519005201198,160520002601128,160520003101298,160520004901104,160522002601180,160522003101181,160522005101066,160523003501137,160524002101384,160525003601139,160527001601331,160529001801044,160529002901143,160529004201340,160529004801032,160529005901303,160530002201097,160530002201257,160530003301125,160530003901058,160530003901333,160530005501141,160530005501252,160531004101070,160612001601085,160613001801242,160723002001090,160811004601305,160812002601261,160812002601305,160812003101071,160812003101329,160813001601121,160813001601145,160813002101166,160813002601010,160813003601035,160813004101210,160816004701066,160817001601061,160817003601196,160916001801332,160919003001171,160919003501265,160919004001388,160921001601375,160923003701120,161006003101374,161006005901017,161006005901093,161007002801226,161009003801069,161012002601039,161104004801070,161105003101341,161105003601351,161105005101023,161106001601034,161106005101129,161107001601024,161108001601107,161108001601161,161108002101119,161108002101187,161108002101191,161108002101276,161109004401299,161115001601321,161115003701356,161116002201129,161117003001373,161117004601128,161117005201359,161118004701017,161118005201319,161118005201320,161119003601228,161119004201024,161119004701107,161211004701345,161212001601261,161212002601295,161213003101278,161213004101025,161213004101339,161213006101249,161217004101382,161217005601184,161217006101204,161218003601063,161218003601339,161219003601037,161219004101152,161228002001366,161229002201030,161229002201157,161229003201274,170102001701165,170103002001235,170103002001282,170103002001305,170104003501083,170105003101209,170105003101352,170106002601102,170106002601197,170106003601362,170106004101174,170107004201003,170108001801317,170108002201097,170108003901230,170109001801137,170109002801135,170109003801174,170109003801273,170111001601016,170112002101382,170114002101063,170114002101339,170114003601290,170114003601376,170114004101016,170115004201185,170118002701030,170118002701196,170119003601220,170120002101312,170121003401169,170121003901077,170122003101340,170122003601090,170122004601377,170128002601033,170128002601165,170129002101137,170129002101347,170130001601230,170130002101033,170130002101165,170130003101174,170130003601123,170130003601297,170131002301087,170131002601060,170202000101092,170202002201185,170202002701140,170203001901377,170205004401331,170206002901377,170206003201058,170206004201046,170206004201362,170216003801367,170217002801068,170217002801198,170217002801380,170217003401017,170217003401147,170218003201051,170219001601009,170220002101343,170220003101314,170220004101091,170220004601158,170314001601103,170403001601136,170404002101149,170404002601205,170404003601229,170406001601159,170408002501011,170408005001067,170408005001308,170410002101348,170410002701272,170410004501313,170411001601075,170411002601025,170411002601191,170411003601098,170411004601220,170412003401378,170412003901309,170412004901309,170412005401011,170412005401071,170413002601076,170413003601390,170413005601362,170414002101052,170414003601101,170414004101073,170414005101017,170415001501228,170415002501228,170415003101175,170415003601025,170415003601189,170415004101174,170416002701066,170416004301114,170416004801007,170416004801173,170416005301236,170417003201089,170418001601275,170418002101090,170418003701290,170506002901361,170506004401240,170506004901324,170507006201080,170507006201193,170507007201143,170507007801208,170507011101008,170507011101311,170508002101378,170508002601032,170508006401160,170509003701142,170509004701227,170509004701322,170510002301338,170510002801322,170510003301004,170510004301135,170510005301033,170510005301334,170511000101285,170511000601034,170511002101319,170511002601236,170511004001142,170512001801232,170513002001161,170513002001248,170513003001056,170513005901113,170513005901274,170514002101234,170514002401391,170515003101149,170515003101382,170515003101384,170516001101192,170516003101009,170517003301089,170530001601095,170530002101034,170531001901293,170531002301126,170531004801028,170531004801031,170531004801294,170531004801322,170531005801077,170601002101044,170601004601088,170602003201109,170602003701127,170602004701040,170602005701036,170602005701175,170603001601041,170603001601240,170603002101001,170604003101383,170604004101246,170604005101083,170604006101133,170614002101091,170614002601258,170614003101292,170614003601197,170614003601379,170615002801275,170615002801378,170615003401208,170615004401292,170615004901182,170710003801342,170711002001178,170711003001299,170711003501376,170711004501079,170711005101326,170712002101108,170712003101052,170712004201168,170712004201252,170712004201380,170712004801373,170712005301253,170713004101294,170723002101172,170723003601071,170724003101397,170724003601373,170725001601153,170725002101149,170725002601369,170725003101359,170725003601264,170801002801335,170801003401153,170801005201398,170802001801207,170805002601032,170806002801288,170806003201171,170806004701008,170829001901364,170829002401364,170829002901058,170905002101267,170906002101297,170907002101224,170907002601319,170907003601087,170908001601222,170909001601119,170910002601388,170910004101195,170910004601040,170911002601399,170911004701034,170911005301285,170912001901244,170912002401194,170912002901077,171001002901207,171003003101183,171004001501223,171027002801191,171027003301191,171031003301012,171031004101383,171101000701097,171102001601319,171102003901026,171104002801315,171106002401004,171106002401398,171205003101203,171206004101388,171206004601182,171206005101197,171206005101201,171207003301095,171207004601047,171208002101228,171208003101051,171208003601388,171227002601086,171227005301347,171228002101301,171228003701253,171230002101062,180101003101029,180101005001256,180102001601305,180102002601265,180102003101022,180102003101216,180102003101233,180102003601223,180102004601331,180102005101317,180103001601109,180103002601216,180103003101314,180129003601025,180129004101115,180129004101357,180129004901318,
    #manual possible cemps
140807005001173,150706004401371,151008002101026,151110001601398,160129005201022,160530002801181,161115002701128,170413005101359,170414002601182,170724002101211,170724004601019,170910005101279,171001001601136
])
print 't-SNE manual points', len(tsne_sel)


# object not in t-SNE
idx_in_tsne = np.in1d(selected_objects['sobject_id'], tsne_sel)
sid_out_blob = selected_objects[~idx_in_tsne]['sobject_id']
# print ','.join([str(s) for s in sid_out_blob])
# raise SystemExit

print 'tSNE no limits:', len(tsne_sel)
# first limit out from tsne members with low swan_fit_integ
# swan_integ_tsne_sel = results_swan_all['sobject_id', 'swan_fit_integ', 'sig'][np.in1d(results_swan_all['sobject_id'], tsne_sel)]
# bad_tsne_sel = swan_integ_tsne_sel['sobject_id'][np.logical_or(swan_integ_tsne_sel['swan_fit_integ'] < 0.2, swan_integ_tsne_sel['sig'] > 1.)]
# tsne_sel = tsne_sel[np.in1d(tsne_sel, bad_tsne_sel, invert=True)]
print 'tSNE limited:', len(tsne_sel)

# investigate inside / outside t-sne detected objects
res_final_temp = join(selected_objects, cannon_param['sobject_id', 'Teff_cannon', 'Logg_cannon', 'Fe_H_cannon', 'flag_cannon'], keys='sobject_id', join_type='left')
print 'Outside tsne blob:', np.sum(np.in1d(integ_sel, tsne_sel, invert=True))
print 'Inside tsne blob common:', np.sum(np.in1d(integ_sel, tsne_sel))
print 'Not in supervised'

sobjects_inside_blob = integ_sel[np.in1d(integ_sel, tsne_sel)]
sobjects_outside_blob = integ_sel[np.in1d(integ_sel, tsne_sel, invert=True)]
# print 'IDS INSIDE:', ','.join([str(si) for si in sobjects_inside_blob])
# print 'IDS OUTSIDE:', ','.join([str(si) for si in sobjects_outside_blob])

# create final selection of objects and append necessary data to it
sobj_final = np.unique(np.hstack((integ_sel, tsne_sel)))
sid_all = galah_data['sobject_id']
sobjects_undetected = sid_all[np.in1d(sid_all, sobj_final, invert=True)]
# print 'IDS RANDOM:', ','.join([str(si) for si in sobjects_undetected[np.random.randint(0,len(sobjects_undetected),250)]])

selected_objects_final = galah_data[np.in1d(galah_data['sobject_id'], sobj_final)]['sobject_id', 'galah_id', 'ra', 'dec', 'teff_guess', 'feh_guess', 'logg_guess', 'red_flag', 'rv_guess', 'e_rv_guess', 'utmjd']
selected_objects_final = join(selected_objects_final, cannon_param['sobject_id', 'Teff_cannon', 'Logg_cannon', 'Fe_H_cannon', 'e_Teff_cannon', 'e_Logg_cannon', 'e_Fe_H_cannon', 'flag_cannon', 'C_abund_cannon', 'flag_C_abund_cannon','Ba_abund_cannon', 'flag_Ba_abund_cannon','O_abund_cannon', 'flag_O_abund_cannon','Li_abund_cannon', 'flag_Li_abund_cannon','Y_abund_cannon','Zr_abund_cannon','Mo_abund_cannon','La_abund_cannon','Ce_abund_cannon','Nd_abund_cannon','Sm_abund_cannon'], keys='sobject_id', join_type='left')
selected_objects_final = join(selected_objects_final, results_swan_all['sobject_id', 'swan_fit_integ'], keys='sobject_id', join_type='left')
print 'CH found in total:', len(sobj_final)
print 'CH found in total:', len(selected_objects_final)
print ''

if Li_LINES:
    # possible Li rich obhects
    print 'Li investigation of objects'
    li_rich = selected_objects_final['sobject_id','Li_abund_cannon','flag_Li_abund_cannon'][selected_objects_final['Li_abund_cannon'] > 1.]
    print ','.join([str(s) for s in li_rich['sobject_id']])
    print li_rich
    print selected_objects_final['sobject_id','Li_abund_cannon','flag_Li_abund_cannon'][selected_objects_final['flag_Li_abund_cannon'] == 0]

    # # investigate s-process elements and their relation to C
    # for s_el in ['Y_abund_cannon','Zr_abund_cannon','Mo_abund_cannon','La_abund_cannon','Ce_abund_cannon','Nd_abund_cannon','Sm_abund_cannon', 'Ba_abund_cannon']:
    #     plt.scatter(selected_objects_final['C_abund_cannon'], selected_objects_final[s_el], lw=0, s=5)
    #     plt.title(s_el)
    #     plt.show()
    #     plt.close()

    # fit a Gaussian or Voigt profile to the Li line
    li_strong = list([])
    for s_id_li in selected_objects_final['sobject_id']:
        print '  Li line fit', s_id_li
        flx, wvl = get_spectra_dr52(str(s_id_li), bands=[3], root=dr53_dir, individual=False, extension=4, read_sigma=False)
        v_model = models.Const1D(amplitude=np.median(flx),
                                 fixed={'amplitude':False}) - \
                  models.Gaussian1D(amplitude=0.3, mean=6707.7635, stddev=0.2,
                                    bounds={'mean':[6707.65, 6707.981], 'stddev':[0.2, 2.]},
                                    fixed={'mean':True})
        fit_t = fitting.LevMarLSQFitter()
        t = fit_t(v_model, wvl[0], flx[0])
        if t.amplitude_1.value >= 0.15:
            print t.amplitude_1.value, t.stddev_1.value
            li_strong.append(s_id_li)
    if len(li_strong) > 0:
        print 'Li strong objects: ', ','.join([str(li_s) for li_s in li_strong])

if H_LINES:
    # fit a Gaussian or Voigt profile to the Li line
    h_depleted = list([])
    for s_id_h in selected_objects_final['sobject_id']:
        print '  H line fit', s_id_h
        flx, wvl = get_spectra_dr52(str(s_id_h), bands=[3], root=dr53_dir, individual=False, extension=4, read_sigma=False)
        H_model = models.Const1D(amplitude=np.median(flx[0]),
                                 fixed={'amplitude':False}) - \
                  models.Gaussian1D(amplitude=0.5, mean=6562.8, stddev=0.2,
                                    bounds={'mean':[6562.6, 6563.0], 'stddev':[0.05, 10.]},
                                    fixed={'mean':True})
        fit_t = fitting.LevMarLSQFitter()
        t = fit_t(H_model, wvl[0], flx[0])
        if t.amplitude_1.value < 0.5:
            print t.amplitude_1.value, t.stddev_1.value
            h_depleted.append(s_id_h)
    if len(h_depleted) > 0:
        print 'H depleted objects: ', ','.join([str(li_s) for li_s in h_depleted])

# out list of objects
txt_out = open(results_folders + 'strong/list_out.txt', 'w')
txt_out.write(','.join([str(s) for s in sobj_final]))
txt_out.close()

# count GALAH stars
g_id = selected_objects_final['galah_id']
g_id = g_id.filled(-1)
print 'Actual GALAH spectra:', np.sum(g_id > 0)

# repeats by coordinates
if REPEATS_ALL:
    print 'Repeats by coordinates between surveys'
    max_sep = 0.5 * un.arcsec
    total_rep = 0
    n_rep = 0
    no_rep = 1
    all_radec = coord.ICRS(galah_data['ra']*un.deg, dec=galah_data['dec']*un.deg)
    all_radec_final = coord.ICRS(selected_objects_final['ra']*un.deg, dec=selected_objects_final['dec']*un.deg)
    rv_diff_all = []
    n_rep_all = []
    all_rv_diff = []
    all_mjd_diff = []
    sid_variable_out = list([])  # used in final table of outputs
    for obj in selected_objects_final:
        obj_radec = coord.ICRS(obj['ra']*un.deg, dec=obj['dec']*un.deg)
        idx_repeats = all_radec.separation(obj_radec) < max_sep
        n_rep_obs = np.sum(idx_repeats)
        idx_repeats_final = coord.ICRS(obj['ra']*un.deg, dec=obj['dec']*un.deg).separation(all_radec_final) < max_sep
        n_rep_obs_final = np.sum(idx_repeats_final)
        if n_rep_obs > 1:
            d_rv = np.max(galah_data[idx_repeats]['rv_guess']) - np.min(galah_data[idx_repeats]['rv_guess'])
            rv_diff_all.append(d_rv)
            n_rep_all.append([n_rep_obs, n_rep_obs_final])
            total_rep += n_rep_obs
            n_rep += 1
            if d_rv > 0.5:
                # only used in final table to mark those objects
                sid_variable_out.append(np.array(galah_data[idx_repeats]['sobject_id']))
        else:
            no_rep += 1
    rv_diff_all, unq_idx = np.unique(rv_diff_all, return_index=True)
    n_rep_all = np.array(n_rep_all)[unq_idx, :]
    perc_n_rep_all = 100.*n_rep_all[:, 1]/n_rep_all[:, 0]

    print 'Perc dete:', 100.*np.sum(n_rep_all[:, 1])/np.sum(n_rep_all[:, 0])
    print ' Unique:', len(rv_diff_all) + no_rep
    print ' N Repeats:', no_rep
    print ' W Repeats:', len(rv_diff_all)
    print 'possible variable (d_rv > 0.25):', np.sum(np.array(rv_diff_all) >= 0.25)
    print 'possible variable (d_rv > 0.50):', np.sum(np.array(rv_diff_all) >= 0.50)
    print 'possible variable (d_rv > 1.00):', np.sum(np.array(rv_diff_all) >= 1.0)
    print 'max d_rv:', np.max(np.array(rv_diff_all))
    print ''

    plt.figure(figsize=(6.4, 3.5))
    log_bins = 10**np.linspace(0., 1.67, 40)-1
    plt.hist(rv_diff_all, range=(0, 46), bins=log_bins, histtype='stepfilled', color='black', alpha=0.2)
    plt.hist(rv_diff_all, range=(0, 46), bins=log_bins, histtype='step', color='black', alpha=1)
    plt.xlabel(r'Maximal RV difference [km s$^{-1}$]')
    plt.ylabel('Number of stars')
    # plt.gca().set_yscale("log", nonposy='clip')
    plt.gca().set_xscale("log", nonposx='clip')
    plt.yticks([0,2.5,5,7.5,10],['0','','5','','10'])
    plt.grid(ls='--', color='black', alpha=0.3)
    plt.tight_layout()
    plt.savefig('rv_rep_dist.png', dpi=300)
    plt.close()

print 'Repeats by coordinates between detected'
max_sep = 0.5 * un.arcsec
total_rep = 0
n_rep = 0
no_rep = 1
all_radec = coord.ICRS(selected_objects_final['ra']*un.deg, dec=selected_objects_final['dec']*un.deg)
rv_diff_all = []
n_rep_all = []
for obj in selected_objects_final:
    obj_radec = coord.ICRS(obj['ra']*un.deg, dec=obj['dec']*un.deg)
    idx_repeats = all_radec.separation(obj_radec) < max_sep
    n_rep_obs = np.sum(idx_repeats)
    if n_rep_obs > 1:
        d_rv = np.max(selected_objects_final[idx_repeats]['rv_guess']) - np.min(selected_objects_final[idx_repeats]['rv_guess'])
        # print ' ', obj['galah_id'], np.array(selected_objects_final[idx_repeats]['sobject_id']), np.array(selected_objects_final[idx_repeats]['rv_guess']), ' d_rv:', d_rv
        rv_diff_all.append(d_rv)
        n_rep_all.append([n_rep_obs, n_rep_obs_final])
        total_rep += n_rep_obs
        n_rep += 1
    else:
        no_rep += 1
rv_diff_all, unq_idx = np.unique(rv_diff_all, return_index=True)
n_rep_all = np.array(n_rep_all)[unq_idx, :]
print ' Unique:', len(rv_diff_all) + no_rep
print ' N Repeats:', no_rep
print ' W Repeats:', len(rv_diff_all)
print 'possible binary (d_rv > 0.25):', np.sum(np.array(rv_diff_all) >= 0.25)
print 'possible binary (d_rv > 0.50):', np.sum(np.array(rv_diff_all) >= 0.50)
print 'possible binary (d_rv > 1.00):', np.sum(np.array(rv_diff_all) >= 1.0)
print 'max d_rv:', np.max(np.array(rv_diff_all))
print ''


# l/b plots
print 'Creating l/b plots of detected objects'
l_b_coords = coord.ICRS(ra=selected_objects_final['ra']*un.deg, dec=selected_objects_final['dec']*un.deg).transform_to(coord.Galactic)
plt.figure()
plt.subplot(111, projection="mollweide")
plt.scatter(np.deg2rad(l_b_coords.l.value)-np.pi, np.deg2rad(l_b_coords.b.value), lw=0, s=2, c='black')
plt.grid(alpha=0.75, ls='--')
plt.tight_layout()
plt.savefig(results_folders + 'strong/l_b.png', dpi=350)
plt.close()


if MOVE_PLOTS:
    for object in selected_objects_final[np.argsort(selected_objects_final['swan_fit_integ'])[::-1]]:  # from strongest to weakest swan fit/enhancement
        s_id = object['sobject_id']
        ra = object['ra']
        dec = object['dec']
        print num, s_id
        id_str = str(s_id)

        cannon_obj_data = cannon_param[cannon_param['sobject_id'] == s_id]
        feh = np.nan
        feh_suf = ''
        if len(cannon_obj_data) == 1:
            feh = cannon_obj_data['Fe_H_cannon'].data[0]
            if cannon_obj_data['flag_cannon'] == 0:
                if feh < -1.:
                    print cannon_obj_data['C_abund_cannon', 'flag_C_abund_cannon']
            else:
                # feh = np.inf
                feh_suf = 'flag'

        input_file = results_folders + str(np.abs(np.int32(s_id/10e10))) + '/' + id_str + '.png'
        if os.path.isfile(input_file):
            os.system('cp '+input_file + ' ' + results_folders + 'strong/{:03.0f}_'.format(num)+id_str+'_ra{:f}_dec{:f}'.format(ra,dec)+'_fehcannon_{:.2f}'.format(feh)+feh_suf+'.png')
        num += 1
        

def get_objects_from_xmatch(files, sobj_analysed):
    s_ids = []
    for f in files:
        f_data = Table.read(f)
        s_ids.append(f_data['sobject_id'].data)
    sids_unique = np.unique(np.hstack(s_ids))
    return sids_unique[np.in1d(sids_unique, sobj_analysed)]


def get_objects_and_feh_xmatch(files, sobj_analysed):
    feh_data = []
    for f in files:
        f_data = Table.read(f)
        f_data = f_data[np.in1d(f_data['sobject_id'], sobj_analysed)]
        if 'e_[Fe/H]' not in f_data.colnames:
            f_data['e_[Fe/H]'] = np.nan
        feh_data.append(f_data['sobject_id', '[Fe/H]', 'e_[Fe/H]'])
    return vstack(feh_data)

sobj_analysed = Table.read(tsne_data_dir + tsne_input_files[0])['sobject_id']
cemps_ref = get_objects_from_xmatch(glob('Reference_lists/xmatch_cemps*.csv'), sobj_analysed)
carbon_ref = get_objects_from_xmatch(glob('Reference_lists/xmatch_carbon*.csv'), sobj_analysed)
cemps_feh = get_objects_and_feh_xmatch(glob('Reference_lists/xmatch_cemps*.csv'), sobj_analysed)

cemps_feh = join(cemps_feh, cannon_param['sobject_id', 'Teff_cannon', 'Fe_H_cannon','e_Fe_H_cannon'], keys='sobject_id')
plt.scatter(cemps_feh['[Fe/H]'], cemps_feh['Fe_H_cannon'], lw=0, s=11, c='black', alpha=0.9)
plt.errorbar(cemps_feh['[Fe/H]'], cemps_feh['Fe_H_cannon'],
             yerr=cemps_feh['e_Fe_H_cannon'],
             xerr=cemps_feh['e_[Fe/H]'],
             c='black', alpha=0.9, fmt='.', elinewidth=0.5)
plt.plot([-3,3], [-3,3], c='black', alpha=0.66, ls='--')
plt.xlabel(r'Published metalicity [M/H]')
plt.ylabel(r'The Cannon [Fe/H]')
plt.xlim(-3.2, -0.35)
plt.ylim(-2.1, 0.35)
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(results_folders + 'strong/cemps_meh_feh.png', dpi=200)
plt.close()

idx_cemp_ref = np.in1d(selected_objects_final['sobject_id'], cemps_ref)
idx_ch_ref = np.in1d(selected_objects_final['sobject_id'], carbon_ref)
print 'Cemps: ', len(cemps_ref), np.sum(idx_cemp_ref)
print 'Carbon:', len(carbon_ref), np.sum(idx_ch_ref)

print 'Cemps ids:', cemps_ref
# print cannon_param[np.in1d(cannon_param['sobject_id'], cemps_ref)]['sobject_id', 'Teff_cannon','Fe_H_cannon', 'Logg_cannon', 'flag_cannon']
# print galah_data[np.in1d(galah_data['sobject_id'], cemps_ref)]['sobject_id', 'teff_guess','feh_guess', 'logg_guess']
print 'Carbon ids:', carbon_ref
# print cannon_param[np.in1d(cannon_param['sobject_id'], carbon_ref)]['sobject_id', 'Teff_cannon','Fe_H_cannon', 'Logg_cannon', 'flag_cannon']
# print galah_data[np.in1d(galah_data['sobject_id'], carbon_ref)]['sobject_id', 'teff_guess','feh_guess', 'logg_guess']


# teff histogram of detected and not detected CH stars
# plt.hist(selected_objects_final['Teff_cannon'], range=[4200,6200], bins=40, alpha=0.2,label='Complete detected set', color='black')
carbon_detected_s = carbon_ref[np.in1d(carbon_ref, selected_objects_final['sobject_id'])]
carbon_undetected_s = carbon_ref[np.in1d(carbon_ref, selected_objects_final['sobject_id'], invert=True)]

fig, ax = plt.subplots(1, 2)
ax[0].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_detected_s)]['Teff_cannon'], range=[4250,6000], bins=30, alpha=1,label='', color='C2', histtype='step')
ax[0].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_detected_s)]['Teff_cannon'], range=[4250,6000], bins=30, alpha=0.3,label='Detected', color='C2', histtype='stepfilled')
ax[0].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_undetected_s)]['Teff_cannon'], range=[4250,6000], bins=30, alpha=1,label='', color='C3', histtype='step')
ax[0].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_undetected_s)]['Teff_cannon'], range=[4250,6000], bins=30, alpha=0.3,label='Undetected', color='C3', histtype='stepfilled')
ax[0].set(xlabel=r'$T_\mathrm{eff}$ [K]', ylabel=r'Number of objects')

ax[1].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_detected_s)]['Logg_cannon'], range=[1,6], bins=30, alpha=1,label='', color='C2', histtype='step')
ax[1].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_detected_s)]['Logg_cannon'], range=[1,6], bins=30, alpha=0.3,label='Detected', color='C2', histtype='stepfilled')
ax[1].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_undetected_s)]['Logg_cannon'], range=[1,6], bins=30, alpha=1,label='', color='C3', histtype='step')
ax[1].hist(cannon_param[np.in1d(cannon_param['sobject_id'], carbon_undetected_s)]['Logg_cannon'], range=[1,6], bins=30, alpha=0.3,label='Undetected', color='C3', histtype='stepfilled')
ax[1].set(xlabel=r'$\log{}$g')

ax[0].legend()
ax[0].grid(ls='--', alpha=0.3)
ax[1].grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(results_folders + 'strong/ch_comb.png', dpi=250)
plt.close()

# print '  low teff: ' + ','.join([str(s) for s in selected_objects_final[selected_objects_final['Teff_cannon']<4650]['sobject_id']])
# print '  high teff:' + ','.join([str(s) for s in selected_objects_final[selected_objects_final['Teff_cannon']>5700]['sobject_id']])


# plot tSNE resulting projections
if TSNE_PLOTS:
    print 'Plotting tSNE projections'
    for tsne_file in tsne_input_files:
        tsne_data = Table.read(tsne_data_dir + tsne_file)
        tsne_data = join(tsne_data, cannon_param['sobject_id', 'Teff_cannon', 'Fe_H_cannon'], keys='sobject_id', join_type='left')

        idx_cemp_ref = np.in1d(tsne_data['sobject_id'], cemps_ref)
        idx_ch_ref = np.in1d(tsne_data['sobject_id'], carbon_ref)
        idx_t = np.in1d(tsne_data['sobject_id'], tsne_sel)
        idx_i = np.in1d(tsne_data['sobject_id'], integ_sel)
        idx_asiago = np.in1d(tsne_data['sobject_id'], [150409005101291])

        plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=0.5, c='#a9a9a9', alpha=0.2, label='')
        # plt.scatter(tsne_data['tsne_axis_1'][idx_t], tsne_data['tsne_axis_2'][idx_t], lw=0, s=0.75, c='r', label='')
        # plt.scatter(tsne_data['tsne_axis_1'][idx_i], tsne_data['tsne_axis_2'][idx_i], lw=0, s=0.75, c='b',label='Supervised')
        # plt.scatter(tsne_data['tsne_axis_1'][idx_asiago], tsne_data['tsne_axis_2'][idx_asiago], lw=0, s=8, c='g', label='Asiago')
        plt.scatter(tsne_data['tsne_axis_1'][idx_ch_ref], tsne_data['tsne_axis_2'][idx_ch_ref], lw=0, s=5, c='g', label='CH')
        plt.scatter(tsne_data['tsne_axis_1'][idx_cemp_ref], tsne_data['tsne_axis_2'][idx_cemp_ref], lw=0, s=5, c='m', label='CEMP')
        #

        plt.xticks([])
        plt.yticks([])
        plt.xlim(-22, 20)
        plt.ylim(-23, 22)
        plt.tight_layout()

        # # cosmetics upgrades
        # ax = plt.gca()
        #
        # from matplotlib.patches import Circle
        # # cemp circles
        # ax.add_patch(Circle((-7.85, -10.2), 0.4, fill=False, lw=0.5, color='black'))
        # ax.add_patch(Circle((3.20, -13.04), 0.3, fill=False, lw=0.5, color='black'))
        #
        # from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        # # zoom for complete clump1
        # axins = zoomed_inset_axes(ax, 6, loc=1)  # zoom = 6
        # axins.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=2., c='#a9a9a9', alpha=0.5)
        # axins.scatter(tsne_data['tsne_axis_1'][idx_t], tsne_data['tsne_axis_2'][idx_t], lw=0, s=2., c='r')
        # axins.scatter(tsne_data['tsne_axis_1'][idx_i], tsne_data['tsne_axis_2'][idx_i], lw=0, s=2., c='b')
        # axins.set(xlim=[-19.3,-17.8], ylim=[1.2,3.8], xticks=[], yticks=[])
        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', ls='--', lw=0.5)
        # # zoom for complete circle 1
        # axins2 = zoomed_inset_axes(ax, 7, loc=3)  # zoom = 6
        # axins2.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=2., c='#a9a9a9', alpha=0.5)
        # axins2.scatter(tsne_data['tsne_axis_1'][idx_t], tsne_data['tsne_axis_2'][idx_t], lw=0, s=2., c='r')
        # axins2.scatter(tsne_data['tsne_axis_1'][idx_i], tsne_data['tsne_axis_2'][idx_i], lw=0, s=2., c='b')
        # axins2.add_patch(Circle((-7.85, -10.2), 0.4, fill=False, lw=0.5, color='black'))
        # axins2.set(xlim=[-8.5, -7.1], ylim=[-10.9, -9.5], xticks=[], yticks=[])
        # mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec='black', ls='--', lw=0.5)
        # # zoom for complete circle 1
        # axins3 = zoomed_inset_axes(ax, 7, loc=4)  # zoom = 6
        # axins3.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=2., c='#a9a9a9', alpha=0.5)
        # axins3.scatter(tsne_data['tsne_axis_1'][idx_t], tsne_data['tsne_axis_2'][idx_t], lw=0, s=2., c='r')
        # axins3.scatter(tsne_data['tsne_axis_1'][idx_i], tsne_data['tsne_axis_2'][idx_i], lw=0, s=2., c='b')
        # axins3.add_patch(Circle((3.20, -13.04), 0.3, fill=False, lw=0.5, color='black'))
        # axins3.set(xlim=[2.5, 3.9], ylim=[-13.7, -12.3], xticks=[], yticks=[])
        # mark_inset(ax, axins3, loc1=2, loc2=3, fc="none", ec='black', ls='--', lw=0.5)

        plt.legend()
        plt.savefig(results_folders + 'strong/' + tsne_file[:-5]+'_refpapers.png', dpi=300)
        plt.close()

    # Cannon parameter plots into selected t-SNE projection
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    # plt.suptitle(r'Spatial distribution of $T_\mathrm{eff}$ [K] and $\mathrm{[Fe/H]}$ in t-SNE projection')
    im1 = ax[0].scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=1., c=tsne_data['Teff_cannon'],
                  vmin=np.percentile(tsne_data['Teff_cannon'], 2.), vmax=np.percentile(tsne_data['Teff_cannon'], 98.))
    ax[0].set(xlim=[-22,20], ylim=[-22,22], xticks=[], yticks=[])#,
              # title=r'Distribution of $T_\mathrm{eff}$ [K] in projection')
    plt.colorbar(im1, ax=ax[0], orientation="horizontal", pad=0.)
    im2 = ax[1].scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=1., c=tsne_data['Fe_H_cannon'],
                  vmin=np.percentile(tsne_data['Fe_H_cannon'], 2.), vmax=np.percentile(tsne_data['Fe_H_cannon'], 98.))
    ax[1].set(xlim=[-22, 20], ylim=[-22, 22], xticks=[], yticks=[])#,
              # title=r'Spatial distribution of $\mathrm{[Fe/H]}$ in projection')
    plt.colorbar(im2, ax=ax[1], orientation="horizontal", pad=0.)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.03, bottom=0.0, right=0.97, top=0.97,
                        wspace=0.05, hspace=0.)
    plt.savefig(results_folders + 'strong/tsne_params_notitle.png', dpi=150)
    plt.close()

if CANNON_PLOTS:
    print 'Cannon analysis'
    c_col = 'C_abund_cannon'
    # create statistics of stellar parameters for selected batch of objects
    idx_cannon_ok = selected_objects_final['flag_cannon'] == 0
    for p_col in ['Teff_cannon', 'Fe_H_cannon', 'Logg_cannon']:
        hist_data = selected_objects_final[p_col][idx_cannon_ok]
        plt.hist(selected_objects_final[p_col], bins=75, range=(np.nanmin(hist_data), np.nanmax(hist_data)), color='black', alpha=1, histtype='step')
        plt.hist(selected_objects_final[p_col], bins=75, range=(np.nanmin(hist_data), np.nanmax(hist_data)), color='black', alpha=0.25, histtype='stepfilled')
        plt.hist(hist_data, bins=75, range=(np.nanmin(hist_data), np.nanmax(hist_data)), color='C0', alpha=1, histtype='step')
        plt.hist(hist_data, bins=75, range=(np.nanmin(hist_data), np.nanmax(hist_data)), color='C0', alpha=0.75, histtype='stepfilled')
        if 'Fe' in p_col:
            plt.axvline(x=-1., c='black', ls='--', alpha=0.8)
            plt.axvline(x=-0.5, c='black', ls='--', alpha=0.3)
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel(r'$\mathrm{[Fe/H]}$')
        plt.ylabel('Number of objects')
        plt.tight_layout()
        plt.savefig(results_folders + 'strong/' + p_col + '.png', dpi=300)
        plt.close()

    # histogram of C abundances
    plt.hist(selected_objects_final[c_col][selected_objects_final['flag_'+c_col] == 0], bins=75)
    plt.axvline(x=1., c='black', ls='--')
    plt.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_folders + 'strong/'+c_col+'.png', dpi=300)
    plt.close()
    idx_strong = selected_objects_final[c_col] >= 1.0
    print selected_objects_final[idx_strong]['sobject_id','Fe_H_cannon', 'flag_cannon', 'C_abund_cannon', 'flag_C_abund_cannon']

    # Keil (aka spectroscopic H-R) diagram
    plt.scatter(selected_objects_final['Teff_cannon'],
                selected_objects_final['Logg_cannon'], lw=0, s=5, c='0.5', alpha=0.75, label='All')
    plt.scatter(selected_objects_final['Teff_cannon'][idx_cannon_ok],
                selected_objects_final['Logg_cannon'][idx_cannon_ok], lw=0, s=7, c='C3', alpha=0.85, label='Unflagged')
    plt.ylim(0.5, 5.5)
    plt.xlim(4300, 5900)
    xteff = np.array([5750,5000])
    plt.plot(xteff, -3./750. * xteff + 25., lw=1, c='C1', ls='--', label='Dwarfs/giants', alpha=0.75)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel(r'$T_\mathrm{eff}$ [K]')
    plt.ylabel(r'$\log g$')
    plt.grid(ls='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(results_folders + 'strong/Kiel_diagram.png', dpi=300)
    plt.close()

    # determine number of possible CEMPs
    idx_med_feh = np.logical_and(selected_objects_final['Fe_H_cannon'] <= -.5, selected_objects_final['flag_cannon'] == 0)
    idx_low_feh = np.logical_and(selected_objects_final['Fe_H_cannon'] <= -1., selected_objects_final['flag_cannon'] == 0)
    idx_low_feh_nf = selected_objects_final['Fe_H_cannon'] <= -1.
    cannon_param_sub_low_feh = selected_objects_final[idx_low_feh]
    idx_strong_c = cannon_param_sub_low_feh[c_col][cannon_param_sub_low_feh['flag_'+c_col] == 0]
    print 'Med feh: ', np.sum(idx_med_feh)
    print 'Low feh: ', np.sum(idx_low_feh)
    print 'Low feh - noflag: ', np.sum(idx_low_feh_nf)
    print selected_objects_final[idx_low_feh_nf]['sobject_id','ra','dec','Fe_H_cannon', 'Teff_cannon', 'Logg_cannon']
    print 'Strong C:', np.sum(idx_strong_c)

# giant/dwarf separation
c_ok = selected_objects_final['flag_cannon'] == 0
y = -3./750. * selected_objects_final['Teff_cannon'][c_ok] + 25.
c_gi = (y - selected_objects_final['Logg_cannon'][c_ok]) > 0
print 'n ok:', np.sum(c_ok)
print 'giants:', np.sum(c_gi)
print 'dwarfs:'
print selected_objects_final['sobject_id','galah_id', 'ra', 'dec','rv_guess','e_rv_guess','Teff_cannon', 'Logg_cannon'][c_ok][~c_gi]
s_id_dwarfs = selected_objects_final['sobject_id'][c_ok][~c_gi]
s_id_giants = selected_objects_final['sobject_id'][c_ok][c_gi]
# selected_objects_final['sobject_id', 'ra', 'dec'][c_ok][~c_gi].write('dwarf_temp.fits')

print 'Repeats dwarfs'
max_sep = 0.5 * un.arcsec
total_rep = 0
n_rep = 0
no_rep = 1
all_radec = coord.ICRS(galah_data['ra']*un.deg, dec=galah_data['dec']*un.deg)
rv_diff_all = []
for obj in selected_objects_final[c_ok][~c_gi]:
    obj_radec = coord.ICRS(obj['ra']*un.deg, dec=obj['dec']*un.deg)
    idx_repeats = all_radec.separation(obj_radec) < max_sep
    n_rep_obs = np.sum(idx_repeats)
    if n_rep_obs > 1:
        d_rv = np.max(galah_data[idx_repeats]['rv_guess']) - np.min(galah_data[idx_repeats]['rv_guess'])
        print ' ', obj['galah_id'], np.array(galah_data[idx_repeats]['sobject_id']), np.array(galah_data[idx_repeats]['rv_guess']), ' d_rv:', d_rv
        rv_diff_all.append(d_rv)
        total_rep += n_rep_obs
        n_rep += 1
    else:
        no_rep += 1
rv_diff_all = np.unique(rv_diff_all)
print ' Unique:', len(rv_diff_all) + no_rep
print ' N Repeats:', no_rep
print ' W Repeats:', len(rv_diff_all)
print 'possible binary (d_rv > 0.25):', np.sum(np.array(rv_diff_all) >= 0.25)
print 'possible binary (d_rv > 0.50):', np.sum(np.array(rv_diff_all) >= 0.50)
print 'max d_rv:', np.max(np.array(rv_diff_all))
print ''

# copy CEMPS and carbons to a new folder
print 'Copying reference objects to new locations'
for s_id in cemps_ref:
    id_str = str(s_id)
    input_file = results_folders + str(np.abs(np.int32(s_id / 10e10))) + '/' + id_str + '.png'
    if os.path.isfile(input_file):
        os.system('cp ' + input_file + ' ' + results_folders + 'cemp/' + id_str + '_'+str(np.sum(np.in1d(selected_objects['sobject_id'], s_id)))+'.png')
for s_id in carbon_ref:
    id_str = str(s_id)
    input_file = results_folders + str(np.abs(np.int32(s_id / 10e10))) + '/' + id_str + '.png'
    if os.path.isfile(input_file):
        os.system('cp ' + input_file + ' ' + results_folders + 'carbon/' + id_str + '_'+str(np.sum(np.in1d(selected_objects['sobject_id'], s_id)))+'.png')

# create a distributable table with results:
# - remove unnecessary cols
# - create marks annotating method of detection
# - filter out bad and mas values
# - mark possible CEMP stars

# Create and export final table that will be published
if FINAL_TABLE:
    from copy import deepcopy
    tab_out = deepcopy(selected_objects_final)

    li_strong = [140118002001247, 140311007101047, 140313005201019, 140314005201348, 140315002501348, 140609001601076,
                 140814003801331, 150830003401142, 160328004701083, 160401005401016, 160529002901347, 160811004601264,
                 161116002201312, 161219004601069, 170206006201265, 170220004101091, 170407005701194, 170412005401071,
                 170413005101072, 170506004901084, 170508006401160, 170515001601352, 170602004701267, 170603003101067,
                 170710002201030, 170723003101366, 170801002501319, 170801003401153, 171004001501223, 171206005101374,
                 180101004301352, 180102001601264]

    # get data about literature matched objects
    lit_dir = '/home/klemen/Carbon-Spectra/Reference_lists/'

    lit_data = {
        # file template 'xmatch_'+csv_file+'.csv'
        'file': ['carbon_alksnis_2001','carbon_christlieb_2001','carbon_ji_2016','cemps_abate_2015','cemps_komiya_2007','cemps_masseron_2010','cemps_placco_2010','cemps_placco_2014','cemps_yoon_2016'],
        'cite': ['2001BaltA..10....1A','2001A&A...375..366C','2016ApJS..226....1J','2015A&A...581A..22A','2007ApJ...658..367K','2010A&A...509A..93M','2010AJ....139.1051P','2014ApJ...797...21P','2016ApJ...833...20Y']
    }
    # extract sobject_ids from matched file lists
    xmatch_obj = list([])
    for csv_file in lit_data['file']:
        xmatch_obj.append(np.array(Table.read(lit_dir+'xmatch_'+csv_file+'.csv')['sobject_id']))
    xmatch_obj = np.unique(np.hstack(xmatch_obj))
    print 'N matched:', len(xmatch_obj)
    print xmatch_obj

    # create a base for the final table
    print 'Creating final table'
    sobj_final = np.unique(np.hstack((integ_sel, tsne_sel, xmatch_obj)))
    tab_out = galah_data[np.in1d(galah_data['sobject_id'], sobj_final)][ 'sobject_id', 'ra', 'dec']
    tab_out = join(tab_out, cannon_param['sobject_id', 'Teff_cannon', 'Logg_cannon', 'Fe_H_cannon', 'e_Teff_cannon', 'e_Logg_cannon', 'e_Fe_H_cannon', 'flag_cannon'],
                                  keys='sobject_id', join_type='left')
    tab_out = join(tab_out, results_swan_all['sobject_id', 'swan_fit_integ'],
                                  keys='sobject_id', join_type='left')

    print 'Loading and adding Gaia ref data'
    galah_gaia = Table.read(galah_data_dir + 'sobject_iraf_53_gaia.fits')
    tab_out = join(tab_out, galah_gaia['sobject_id', 'source_id'], keys='sobject_id', join_type='left')

    print 'Current cols in table:', tab_out.colnames

    # rename cols
    tab_out['Teff_cannon'].name = 'teff'
    tab_out['e_Teff_cannon'].name = 'e_teff'
    tab_out['Logg_cannon'].name = 'logg'
    tab_out['e_Logg_cannon'].name = 'e_logg'
    tab_out['Fe_H_cannon'].name = 'feh'
    tab_out['e_Fe_H_cannon'].name = 'e_feh'
    tab_out['flag_cannon'].name = 'flag_cannon'
    tab_out['swan_fit_integ'].name = 'swan_integ'

    # add detection flags
    tab_out['det_sup'] = False
    tab_out['det_usup'] = False
    tab_out['li_strong'] = False
    tab_out['det_sup'][np.in1d(tab_out['sobject_id'], integ_sel)] = True
    tab_out['det_usup'][np.in1d(tab_out['sobject_id'], tsne_sel)] = True
    tab_out['li_strong'][np.in1d(tab_out['sobject_id'], li_strong)] = True

    # add other cols
    tab_out['type'] = '-'
    tab_out['type'][np.in1d(tab_out['sobject_id'], s_id_dwarfs)] = 'D'
    tab_out['type'][np.in1d(tab_out['sobject_id'], s_id_giants)] = 'G'

    # add cemp candidates and rv variable stars
    tab_out['rv_var'] = False
    tab_out['cemp_cand'] = False
    idx_cemp_cand = np.logical_and(tab_out['feh'] <= -1.,
                                   np.in1d(tab_out['sobject_id'], np.unique(np.hstack((integ_sel, tsne_sel)))))
    print 'CEMP cands:', np.sum(idx_cemp_cand)
    tab_out['cemp_cand'][idx_cemp_cand] = True
    if 'sid_variable_out' in locals():
        sid_variable_out = np.unique(np.hstack(sid_variable_out))
        idx_rvvar = np.in1d(tab_out['sobject_id'], sid_variable_out)
        print 'Variable sids:', np.sum(idx_rvvar)
        tab_out['rv_var'][idx_rvvar] = True

    # add literature col
    tab_out['bib_code'] = '------------------------------------------------------------'
    tab_out['bib_code'] = '-'
    for i_csv, csv_file in enumerate(lit_data['file']):
        xmatch_sid = Table.read(lit_dir+'xmatch_'+csv_file+'.csv')['sobject_id']
        xmatch_sid = np.unique(xmatch_sid)
        # add literature string to the matches
        for x_sid in xmatch_sid:
            idx_out = np.where(tab_out['sobject_id'] == x_sid)[0]
            if len(idx_out) != 1:
                continue
            idx_out = idx_out[0]
            if tab_out['bib_code'][idx_out] == '-':
                tab_out['bib_code'][idx_out] = lit_data['cite'][i_csv]
            else:
                tab_out['bib_code'][idx_out] = tab_out['bib_code'][idx_out]+' '+lit_data['cite'][i_csv]

    # drop useless cols
    # tab_out.remove_columns(['sobject_id'])

    # reorder cols and save table
    print 'Final cols in table:', tab_out.colnames
    print 'Final nuber of rows:', len(tab_out)

    # write out results
    tab_out_cols_order = ['source_id', 'sobject_id', 'ra', 'dec', 'det_sup', 'det_usup', 'swan_integ',
                          'teff', 'e_teff', 'logg', 'e_logg', 'feh', 'e_feh', 'flag_cannon',
                          'type', 'rv_var', 'li_strong', 'cemp_cand', 'bib_code']
    tab_out = tab_out[np.argsort(tab_out['ra'])]
    tab_out = tab_out[tab_out_cols_order]
    tab_out.write('galah_carbon_cemp.fits', overwrite=True)

    # CSV export
    csv_out = open('galah_carbon_cemp.csv', 'w')
    csv_out.write(','.join([str(cn) for cn in tab_out.colnames])+'\n')
    for td in tab_out:
        str_out = ','.join([str(cd) for cd in td])
        str_out = str_out.replace('--', 'nan')
        csv_out.write(str_out+'\n')
    csv_out.close()

