""" Loads program configuration into a config object."""

import configparser 


class Config(object):
    def __init__(self, filename):
        self.load_config(filename)
        
 
    def load_config(self, filename):
        config = configparser.SafeConfigParser()
        config.read(filename)
        self.config = config

        s = "general"
        self.name = config.get(s, 'name')
        self.device = []

        s = "noise"
        self.snr = config.getfloat(s, 'snr')
        self.distribution = config.get(s, 'distribution')
        
        s = "blur"
        self.blurring = config.getboolean(s, 'blurring')
        self.sigma = config.getfloat(s, 'sigma')
        self.rate = config.getfloat(s, 'rate')
    
        s = "generator"
        self.VolumeSize = config.getint(s, 'VolumeSize')
        self.ProjectionSize = config.getint(s, 'ProjectionSize')        

        s = "discriminator"
        self.Dtype = config.getint(s, 'Dtype')
        self.leak_value = config.getfloat(s, 'leak_value')
        self.Lambda = config.getfloat(s, 'Lambda')
        self.bias = config.getboolean(s, 'bias')            
        self.num_channel_Discriminator=config.getint(s, 'num_channel_Discriminator')
        self.num_layer_Discriminator=config.getint(s, 'num_layer_Discriminator')
        self.num_N_Discriminator=config.getint(s, 'num_n_discriminator')

        s = "optimization"
        self.epochs=config.getint(s, 'epochs')
        self.batch_size = config.getint(s, 'batch_size')
        self.lambdaPenalty=config.getfloat(s, 'lambdaPenalty')            
        self.gamma_gradient_penalty=config.getfloat(s,'gamma_gradient_penalty')                        
        self.step_size = config.getfloat(s, 'step_size')
        self.gamma = config.getfloat(s, 'gamma')

        
        s = "optimization_gen"
        self.gen_optimizer = config.get(s, 'gen_optimizer')
        self.gen_lr = config.getfloat(s, 'gen_lr')
        self.gen_momentum = config.getfloat(s, 'gen_momentum')
        self.gen_beta_1=config.getfloat(s, 'gen_beta_1')
        self.gen_beta_2=config.getfloat(s, 'gen_beta_2')
        self.gen_eps=config.getfloat(s, 'gen_eps')
        self.gen_clip_grad=config.getboolean(s, 'gen_clip_grad')
        self.gen_clip_norm_value=config.getfloat(s, 'gen_clip_norm_value')
        self.gen_weight_decay=config.getfloat(s, 'gen_weight_decay')

        s = "optimization_dis"
        self.dis_iterations = config.getint(s, 'dis_iterations')
        self.dis_optimizer = config.get(s, 'dis_optimizer')
        self.dis_lr = config.getfloat(s, 'dis_lr')
        self.dis_beta_1=config.getfloat(s, 'dis_beta_1')
        self.dis_beta_2=config.getfloat(s, 'dis_beta_2')
        self.dis_eps=config.getfloat(s, 'dis_eps')
        self.dis_clip_grad=config.getboolean(s, 'dis_clip_grad')
        self.dis_clip_norm_value=config.getfloat(s, 'dis_clip_norm_value')
        self.dis_weight_decay=config.getfloat(s, 'dis_weight_decay')
        




 
        

        

