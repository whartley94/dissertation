from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=10000, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=5, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--update_html_freq', type=int, default=10000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--lambda_GAN', type=float, default=0., help='weight for GAN loss')
        parser.add_argument('--lambda_A', type=float, default=1., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=1., help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                            'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--avg_loss_alpha', type=float, default=.986, help='exponential averaging weight for displaying loss')
        parser.add_argument('--data_dir', type=str, default='/home/s1843503/datasets/INetData/Torr/Tiny/', help='root folder of dataset to use')
        parser.add_argument('--invisible_network', type=bool, default=False, help='cancel print network layers on init')
        parser.add_argument('--save_npy', type=bool, default=True, help='Save numpy arrays during training')
        parser.add_argument('--save_mpl', type=bool, default=True, help='Save png loss log during training')
        parser.add_argument('--auto_names', type=list, default=['slurm_id', 'gpu_ids', 'batch_size', 'num_threads', 'phase', 'load_sg_model'])
        parser.add_argument('--slurm_id', type=int, default=0)
        parser.add_argument('--weighted_mask', action='store_true', help='Do my version of training with weighted masks')
        parser.add_argument('--bb_mask', action='store_true', help='Do my version of training with bounding box masks')
        parser.add_argument('--plot_data_gen', action='store_true', help='Plot data as its generated')
        parser.add_argument('--save_optimizer', action='store_true', help='Save optimiser states when saving model')
        self.isTrain = True
        return parser
