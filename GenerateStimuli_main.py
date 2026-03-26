import os
os.chdir('/run/media/medathati/4TBInt/MedathatiExt/Work/OpticalFlow/Code/InteractiveResearchTool/')

from GenerateMovingStimuli import *
import itertools

import yaml
def load_config():
    with open("dataset_config_base.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg
cfg = load_config()
print(cfg)    



if __name__ == '__main__':
 
    Width = 1024 # 1280
    Height = 436    # 768
    cfg['im_size'] = {'Height':Height ,'Width':Width}
    #Results_base_dir = '/run/media/medathati/4TBInt/MedathatiExt/Work/OpticalFlow/Data/Stimuli/Hyderabad/'
    cfg['base_dir'] = '/run/media/medathati/4TBInt/MedathatiExt/Work/OpticalFlow/Data/Stimuli/Hyderabad/'
    
    
    '''Rectangles'''
    vel_tsample = np.logspace(0.1, np.log10(25), 8, endpoint=True)
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    velocities  = itertools.combinations_with_replacement(vel_sample,2)
    cfg['rectangles']['params']['velocities'] = velocities

    orientations = np.linspace(0,180,7)
    orientations = orientations[:-1]
    cfg['rectangles']['params']['orientations'] = orientations
    
    lengths = [10,25,50]
    cfg['rectangles']['params']['lengths'] = lengths

    aspect_ratios = [0.5,1.0,1.5]
    cfg['rectangles']['params']['aspect_ratios'] = aspect_ratios

    for l in lengths:
        for asr in aspect_ratios:
            GenerateMovingRectangles(cfg['im_size']['Height'], cfg['im_size']['Width'],l,asr,orientations, vel_sample,cfg['base_dir'])
    
    with open(r'Hyderabad_dataset_config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    '''*'''
    
    ''' Circles'''
    ### Generate Circular aperature stimuli
    
    cfg['circles']['params']['radii'] = np.array([21,42])
    cfg['circles']['params']['isfilled'] = ['True','False']

    vel_tsample = np.logspace(0.1, np.log10(25), 8, endpoint=True)
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    velocities  = itertools.combinations_with_replacement(vel_sample,2)
    cfg['circles']['params']['velocities'] = velocities

    for circ_radius in cfg['circles']['params']['radii']:
        print("Generating moving circles with ",circ_radius)
        GenerateMovingCircle(Height, Width,circ_radius,'perimeter', vel_sample,cfg['base_dir'])
        GenerateMovingCircle(Height, Width,circ_radius,'full', vel_sample,cfg['base_dir'])
    
    with open(r'Hyderabad_dataset_config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    '''*'''

    '''Lines'''
    cfg['lines']['params']['lengths'] = np.array([10,20,30])
    orientations = np.linspace(0,180,7)
    orientations = orientations[:-1]
    cfg['lines']['params']['orientations'] = orientations

    vel_tsample = np.logspace(0.1, np.log10(25), 8, endpoint=True)
    vel_sample = np.round(np.concatenate((-1*vel_tsample[::-1],[0],vel_tsample),axis=0)).astype(int)
    velocities  = itertools.combinations_with_replacement(vel_sample,2)
    cfg['lines']['params']['velocities'] = velocities

    for line_length in cfg['lines']['params']['lengths']:
        print("Generating moving lines with ",line_length)
        GenerateMovingLine(cfg['im_size']['Height'], cfg['im_size']['Width'],line_length,orientations, vel_sample,cfg['base_dir'])
    '''*'''

    '''Gratings'''
    cfg['gratings']['params']['aperture_radii'] = np.array([50,75,100,125])
    orientations = np.linspace(0,180,7)
    orientations = orientations[:-1]
    cfg['gratings']['params']['orientations'] = orientations

    for Aperture_Radius in cfg['gratings']['params']['aperture_radii']:
        print("Generating moving gratings with ",Aperture_Radius)
        GenerateMovingGratings(cfg['im_size']['Height'], cfg['im_size']['Width'],Aperture_Radius, orientations, cfg['base_dir']) 
    '''*'''
    

    '''Plaids'''
    frequencies = [1.0/10,1.0/20,1.0/30] 
    cfg['plaids']['params']['frequencies '] = frequencies 
    cfg['plaids']['params']['aperture_radii'] = np.array([100])
    orientations = np.linspace(0,180,7)
    orientations = orientations[:-1]
    cfg['plaids']['params']['orientations'] = orientations
    orientation_offset = np.linspace(30.0,90.0,5)
    orientation_offset = orientation_offset[np.nonzero(orientation_offset)]
    cfg['plaids']['params']['orientation_offset'] = orientation_offset

    #frequencies = [1.0/3,1.0/5,1.0/7,1.0/10,1.0/20,1.0/30] 
    # #Aperture_Radius = 100    
    GenerateMovingPlaids(cfg['im_size']['Height'], cfg['im_size']['Width'],cfg['plaids']['params']['aperture_radii'][0], cfg['plaids']['params']['frequencies '],orientations,orientation_offset, cfg['base_dir'])
    

    '''Plaids'''
    frequencies = [1.0/10,1.0/20,1.0/30] 
    cfg['hybridplaids']['params']['frequencies '] = frequencies 
    cfg['hybridplaids']['params']['aperture_radii'] = np.array([100])
    orientations = np.linspace(0,180,7)
    orientations = orientations[:-1]
    cfg['hybridplaids']['params']['orientations'] = orientations
    orientation_offset = np.linspace(30.0,90.0,5)
    orientation_offset = orientation_offset[np.nonzero(orientation_offset)]
    cfg['hybridplaids']['params']['orientation_offset'] = orientation_offset
    cfg['hybridplaids']['params']['gratings_offset']  = 50

    #frequencies = [1.0/3,1.0/5,1.0/7,1.0/10,1.0/20,1.0/30] 
    # #Aperture_Radius = 100

    GenerateMovingHybridPlaids(cfg['im_size']['Height'], cfg['im_size']['Width'],cfg['plaids']['params']['aperture_radii'][0],cfg['plaids']['params']['frequencies '],orientations, orientation_offset,cfg['hybridplaids']['params']['gratings_offset'], cfg['base_dir'])

    '''Barber Poles'''
    bp_rect_lengths = [200, 300]
    bp_rect_widths = [40, 60]
    bp_frequencies = [1.0/10, 1.0/20, 1.0/30]
    bp_grating_orientations = np.linspace(0, 180, 7)[:-1]
    bp_aperture_orientations = [0, 45, 90, 135]
    cfg['barberpoles'] = {'params': {
        'rect_lengths': bp_rect_lengths,
        'rect_widths': bp_rect_widths,
        'frequencies': bp_frequencies,
        'grating_orientations': list(bp_grating_orientations),
        'aperture_orientations': bp_aperture_orientations
    }}

    for rl in bp_rect_lengths:
        for rw in bp_rect_widths:
            print("Generating barber poles with length=%d, width=%d" % (rl, rw))
            GenerateMovingBarberPole(Height, Width, rl, rw, bp_frequencies,
                                     bp_grating_orientations, bp_aperture_orientations, cfg['base_dir'])

    '''Barber Plaids'''
    bp_orientation_offset = np.linspace(30.0, 90.0, 5)
    bp_orientation_offset = bp_orientation_offset[np.nonzero(bp_orientation_offset)]
    cfg['barberplaids'] = {'params': {
        'rect_lengths': bp_rect_lengths,
        'rect_widths': bp_rect_widths,
        'frequencies': bp_frequencies,
        'orientations': list(bp_grating_orientations),
        'orientation_offset': list(bp_orientation_offset),
        'aperture_orientations': bp_aperture_orientations
    }}

    for rl in bp_rect_lengths:
        for rw in bp_rect_widths:
            print("Generating barber plaids with length=%d, width=%d" % (rl, rw))
            GenerateMovingBarberPlaid(Height, Width, rl, rw, bp_frequencies,
                                      bp_grating_orientations, bp_orientation_offset,
                                      bp_aperture_orientations, cfg['base_dir'])

    '''Transparent Motion (Random Dots)'''
    tm_num_dots = [200, 400]
    tm_dot_radii = [2, 4]
    tm_speed_pairs = [(3, 5), (5, 5), (3, 10), (8, 8)]
    tm_direction_pairs = [(0, 180), (0, 90), (45, 225), (30, 150), (0, 120), (60, 300)]
    cfg['transparent_dots'] = {'params': {
        'num_dots': tm_num_dots,
        'dot_radii': tm_dot_radii,
        'speed_pairs': tm_speed_pairs,
        'direction_pairs': tm_direction_pairs
    }}

    for nd in tm_num_dots:
        for dr in tm_dot_radii:
            print("Generating transparent motion dots with n=%d, r=%d" % (nd, dr))
            GenerateTransparentMotion(Height, Width, nd, dr, tm_speed_pairs,
                                      tm_direction_pairs, cfg['base_dir'])

    '''Transparent Motion (Overlapping Gratings)'''
    tg_frequencies = [1.0/10, 1.0/20, 1.0/30]
    tg_orientations = np.linspace(0, 180, 7)[:-1]
    tg_orientation_offset = np.linspace(30.0, 90.0, 5)
    tg_orientation_offset = tg_orientation_offset[np.nonzero(tg_orientation_offset)]
    tg_radius = 100
    cfg['transparent_gratings'] = {'params': {
        'frequencies': tg_frequencies,
        'orientations': list(tg_orientations),
        'orientation_offset': list(tg_orientation_offset),
        'aperture_radius': tg_radius
    }}
    print("Generating transparent gratings with R=%d" % tg_radius)
    GenerateTransparentGratings(Height, Width, tg_radius, tg_frequencies,
                                 tg_orientations, tg_orientation_offset, cfg['base_dir'])

    with open(r'Hyderabad_dataset_config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    vel_tsample = np.logspace(0.1, np.log10(25), 8, endpoint=True)
    VisualizeVelocitySamples(Height, Width,vel_tsample)
    