Templates
=========


wrkflowa
--------

.. code-block:: python

    from lwf.utils import WorkFlow
    from lwf.experiment import PipeLine, get_ppls
    from typing import List, Callable, Dict, Any, Union, Optional, cast
    from pathlib import Path

    import pandas as pd
    from copy import deepcopy
    import torch
    import cv2
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from torchvision import transforms

    from PIL import Image

    import psutil

    class PaTSR(WorkFlow):
        def __init__(self, loc = None):
            super().__init__(loc)
            self.args = {'num_epochs'}
            self.paths = {"weight.weight", "history.train","history.val", 'pred.pred', "state", "quick"}
            self.template = {
                'dataset', "model", "optimizer",  'val_metric', "iloss", "early_stopper", 'scheduler',
                'train_data_src', 'val_data_src', 'train_batch_size', 'val_batch_size', 'pred_src'
            }
            self.logings = {
                "history.train": ['iter', 'epoch', 'iloss'],
                "history.val": ['epoch', 'iloss', 'brisque', 'ssim', 'psnr', 'lpips', 'train_iloss']
            }

        def _setup(self, args):
            self.num_epochs = args['num_epochs']
            self.comps = { }
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def run(self):
            for current_epoch in range(self.current_epoch, self.num_epochs):
                if not self.P.should_running or self.comps['early_stopper'].early_stop:
                    self.save_state()
                    if self.comps['early_stopper'].early_stop:
                        print("training finished")
                    return

                self.train(current_epoch=current_epoch)

        def _adjust_loader_params(self, mode: str, args: Optional[dict] = None) -> dict:

            args = self.args if args is None else args
            loc = args["dataset"]["loc"]
            dsargs = args["dataset"]["args"]

            if mode in {"val", "train"}:
                dsargs["data_src"] = args[f"{mode}_data_src"]
                ds = self.load_component(loc=loc, args=dsargs)
                collate_fn = getattr(ds, "collate_fn", None) or None
                batch_size = args[mode + "_batch_size"]
                shuffle = not mode == "val  "
            else:
                raise ValueError(mode + "_data_src is not found")

            num_cpu_cores = os.cpu_count()

            if len(ds) < batch_size:
                batch_size = len(ds)
                print(
                    "Warning: Dataset size is smaller than the batch size."
                    f"Adjusting batch size to {batch_size}."
                )
                args.update({mode + "_batch_size": batch_size})
                if self.args is not None:
                    self._save_config()

            pin_memory = batch_size >= 32  # Larger batches benefit more from pin_memory

            if batch_size < 16:
                num_workers = max(1, num_cpu_cores // 2)  # Fewer workers for small batches
            elif batch_size < 64:
                num_workers = num_cpu_cores
            else:
                num_workers = min(num_cpu_cores * 2, 16)

            system_memory_available = psutil.virtual_memory().available > 5 * 1024**3
            if not system_memory_available:
                num_workers = min(num_workers, 4)
                pin_memory = False  # Disable pin_memory to save memory
                print(
                    f"memory available={psutil.virtual_memory().available}<={5 * 1024**3}"
                    " --> pin_memory={pin_memory}"
                )

            return {
                "dataset": ds,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "collate_fn": collate_fn,
                "pin_memory": pin_memory,
                "persistent_workers": True  # workers stay alive across epochs
            }
    
        def _setup_dataloaders(self, args):

            self.trainDataLoader = DataLoader(
                **self._adjust_loader_params(mode="train", args=args)
            )
            self.validDataLoader = DataLoader(
                **self._adjust_loader_params(mode="val", args=args)
            )
    
        def prepare(self):
            if not self.P.cnfg:
                print("not initiated")
                return
            # print(self.P.pplid)
            args = deepcopy(self.P.cnfg["args"])

            self.comps["model"] = self.load_component(**args["model"]).to(self.device) 
            
            args["optimizer"]["args"]["model_parameters"] = self.comps["model"].parameters()

            self.comps["optimizer"] = self.load_component(**args["optimizer"])

            self.comps["early_stopper"] = self.load_component(**args['early_stopper'])

            args['scheduler']['args']['optimizer'] = self.comps["optimizer"].optimizer
            self.comps["scheduler"] = self.load_component(**args['scheduler'])
            
            self.comps['val_metric'] = self.load_component(**args['val_metric'])

            self.comps["iloss"] = self.load_component(**args["iloss"]).to(self.device)

            self._setup_dataloaders(args=args)

            self.resume()
            print("Data loaders are successfully created")

            return True
        
        def resume(self):
            with open(self.P.get_path(of="quick"), encoding="utf-8") as quick:
                quick = json.load(quick)
            self.current_epoch = quick['last']['epoch'] if quick['last']['epoch'] else 0
            self.current_step = quick['last']['iter'] if quick['last']['iter'] else 0
            
            model_path = self.P.get_path(of="weight.weight", args={'epoch': self.current_epoch} )
            
            if not os.path.exists(model_path) and self.current_epoch >0:
                model_path = self.P.get_path(of="weight.weight", args={'epoch': self.current_epoch - 1} )
            
            if os.path.exists(model_path):
                self.comps["model"].load_state_dict(torch.load(model_path))
            
            checkpoint = self.P.get_path(of='state')
            if self.current_epoch > 0 and os.path.exists(checkpoint):
                checkpoint = torch.load(checkpoint)
                # Restore each component
                self.comps['scheduler'].load_state_dict(checkpoint['scheduler'])
                self.comps['optimizer'].load_state_dict(checkpoint['optimizer'])
                self.comps['early_stopper'].load_state(checkpoint['early_stopper'])  # custom load\

        def train(self, current_epoch):
            current_step = self.current_step
            self.comps['model'].train()
            
            total_loss = 0.0
            current_epoch += 1
            for lr, hr, seg in self.trainDataLoader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                seg = seg.to(self.device)
                current_step +=1

                up, sr = self.comps['model'](lr)

                # Compute loss
                loss = self.comps['iloss'](sr, hr, up, seg)

                # Backward
                self.comps['optimizer'].zero_grad()
                loss.backward()

                self.comps['optimizer'].step()

                total_loss += loss.item()

                data = {
                        "iter": current_step, 
                        "epoch": current_epoch,
                        "iloss": loss.item()
                    }

                self.log(of="history.train",data = data)

            avg_train_loss = total_loss / len(self.trainDataLoader)

            self.log(of="quick", data = {'last':data})
            self.log(of="weight.weight", data={ "epoch":current_epoch})
            data = self.val(epoch=current_epoch)
            data['train_iloss'] = avg_train_loss
            self.log(of="history.val",data = data)
            
            self.current_step = current_step
            
        def new( self, args: Dict[str, Any]) -> None:
            if not self.template.issubset(set(args.keys())):
                raise ValueError(f'the args should have {", ".join(self.template- set(list(args.keys())))}')
            
            for i in self.logings:
                record = pd.DataFrame([], columns=self.logings[i])
                record.to_csv(self.P.get_path(of=i), index=False)

            quick = {
                "last": {
                    'epoch':0, 'iter':0
                },
                "best": {
                    
                }
                }
            with open(
                self.P.get_path(of="quick", pplid=self.P.pplid), "w", encoding="utf-8"
            ) as out_file:
                json.dump(quick, out_file, indent=4)

        def log(self, of:str, data:dict):
            if "history" in of:
                metrics = self.logings[of]
                record = pd.DataFrame([[data[i] for i in metrics]], columns=metrics)
                record.to_csv(
                    self.P.get_path(of=of),
                    mode="a",
                    header=False,
                    index=False,
                )
            elif "weight.weight" == of:
                torch.save(
                        self.comps["model"].state_dict(),
                        self.P.get_path(of=of, args=data),
                    )
            elif 'pred' in of:
                save_img_path = self.P.get_path(of = 'pred.gpred', epoch = data['epoch'], args = {'idx':data['idx'], 'epoch':data['epoch']})
                cv2.imwrite(save_img_path, data['sr_img'])
            elif of=='quick':
                with open(self.P.get_path(of='quick')) as fl:
                    qck = json.load(fl)
                qck.update(data)
                # Write back to the same file
                with open(self.P.get_path(of='quick'), 'w') as fl:
                    json.dump(qck, fl, indent=4)
        
        def get_path(self, of: str, pplid: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> str:
            
            pplid = pplid or self.pplid
            if not pplid:
                raise ValueError("Experiment ID (pplid) must be provided.")
            if self.P is None:
                raise ValueError("it dont have a pipeline")

            elif "weight" in of:
                epoch = args.get('epoch', None)
                if epoch is None:
                    
                    raise ValueError(
                        "Epoch must be specified or defined in config under 'best.epoch'."
                    )
                path = Path(*of.split(".")) / pplid / f"{pplid}_e{epoch}.pt"

            elif "history" in of:
                path = Path(*of.split(".")) / f"{pplid}.csv"
                
            elif "pred" in of:
                epoch = args.get('epoch', None)
                if epoch is None:
                    raise ValueError(
                        "Epoch must be specified."
                    )
                if of=='pred.pred':
                        path = Path(*of.split(".")) / pplid / f"e{epoch}"
                else:
                        raise ValueError(
                            f"Invalid value for 'of': {of}. Supported values: "
                            "'config', 'weight', 'history', 'quick'."
                        )
            elif "state" == of:
                path = Path("states") / f"{self.P.pplid}.pth"
            elif of == "quick":
                path = Path("Quicks") / f"{pplid}.json"
            else:
                raise ValueError(
                    f"Invalid value for 'of': {of}. Supported values: "
                    "'config', 'weight', 'gradient', 'history', 'quick'."
                )
            
            return path
        
        def val(self, epoch) -> tuple:
            avg_data = {i:0 for i in self.logings['history.val']}
            self.comps['model'].eval()
            for idx, (lr, hr, seg) in enumerate(self.validDataLoader):

                lr = lr.to(self.device)
                hr = hr.to(self.device)
                seg = seg.to(self.device)

                
                with torch.no_grad():
                up, sr = self.comps['model'](lr)

                iloss = self.comps['iloss'](sr, hr, up, seg)

                data = self.comps['val_metric'](sr, hr)


                data['iloss'] = iloss.item()

                for i in data:
                    avg_data[i] += data[i]
                
                # if (idx+1)%7==0:
                #     print('testinggg..')
                #     break

            avg_data = {i:avg_data[i]/idx for i in avg_data} 

            avg_data['epoch'] = epoch

            self.comps['early_stopper'].step(avg_data)
            self.comps['scheduler'].step(avg_data)
            # self.log(of="history.val", data=avg_data)
            input_dir = self.P.cnfg['args']['pred_src']
            output_dir = self.P.get_path(of="pred.pred", args={'epoch':epoch})
                    # save fake img  to a given path
            self.run_inference(input_dir=input_dir,output_dir=output_dir)
            return avg_data
        
        def get_state(self):
            state = {
                'scheduler': self.comps['scheduler'].state_dict(),
                'optimizer': self.comps['optimizer'].state_dict(),
                'early_stopper': self.comps['early_stopper'].state_dict()

                }
            return state

        def save_state(self):
            state = self.get_state()
            torch.save(state, self.P.get_path(of='state'))

        def run_inference(self, input_dir, output_dir ):
            os.makedirs(output_dir, exist_ok=True)

            device = next(self.comps['model'].parameters()).device

            # Transform: convert PIL to tensor
            size = self.P.cnfg['args']['dataset']['args']['size']
            scale = self.P.cnfg['args']['dataset']['args']['degrade_scale']
            
            trans = transforms.Compose([
                transforms.CenterCrop(size//scale),
                transforms.ToTensor()
            ])

            img_tensors = []
            img_filenames = []

            # Load and transform all images
            for fl in os.listdir(input_dir):
                if not fl.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue  # Skip non-image files

                img_path = os.path.join(input_dir, fl)
                img = Image.open(img_path).convert("L")
                img_tensor = trans(img)  # [C, H, W]
                img_tensors.append(img_tensor)
                img_filenames.append(fl)

            if not img_tensors:
                print("No valid images found.")
                return

            # Stack all into a single batch tensor: [B, C, H, W]
            batch = torch.stack(img_tensors).to(device)

            # Inference
            with torch.no_grad():
                _, fakes = self.comps['model'](batch)  # [B, C, H, W]

            # Save each output image
            for fake, fname in zip(fakes, img_filenames):
                output_path = os.path.join(output_dir, fname)
                save_image(fake, output_path)

        def clean(self):
            h = ["psnr", 'epoch']
            df = pd.read_csv(self.P.get_path(of='history.val'))
            epochs = []
            for col in df.columns:
                if col in h:
                    # get the row index where column is max
                    idx = df[col].idxmax()
                else:
                    # get the row index where column is min
                    idx = df[col].idxmin()
                # get the 'epoch' value at that row
                epochs.append(int(df.loc[idx, 'epoch']))

            for e in range(1, max(epochs)+1):
                if e not in epochs:
                    pth = self.P.get_path(of="weight.weight", args={'epoch':e})
                    if os.path.exists(pth):
                        os.remove(pth)

        def status(self):
            df = pd.read_csv(self.P.get_path(of='history.val'))
            if not df.empty:
                idx = df['epoch'].max()
                data = {'epochs': idx}
            else:
                data = {'epochs': 0}
            return data


    class TestPAtsr(WorkFlow):
        def __init__(self):
            super().__init__()
            self.args = {}
            self.paths = {'logit.pred'}
            self.template = {
                'ppild', 'data_src', 'dataset', 'epoch'
            }
        
        def _setup(self,args):
            self.comps = { }
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _adjust_loader_params(self, args: Optional[dict] = None) -> dict:

            args = self.args if args is None else args
            loc = args["dataset"]["loc"]
            dsargs = args["dataset"]["args"]


            dsargs["data_src"] = args["data_src"]
            ds = self.load_component(loc=loc, args=dsargs)
            collate_fn = getattr(ds, "collate_fn", None) or None
            batch_size = args["batch_size"]
            shuffle = False

            num_cpu_cores = os.cpu_count()

            pin_memory = batch_size >= 32  # Larger batches benefit more from pin_memory

            if batch_size < 16:
                num_workers = max(1, num_cpu_cores // 2)  # Fewer workers for small batches
            elif batch_size < 64:
                num_workers = num_cpu_cores
            else:
                num_workers = min(num_cpu_cores * 2, 16)

            system_memory_available = psutil.virtual_memory().available > 5 * 1024**3
            if not system_memory_available:
                num_workers = min(num_workers, 4)
                pin_memory = False  # Disable pin_memory to save memory
                print(
                    f"memory available={psutil.virtual_memory().available}<={5 * 1024**3}"
                    " --> pin_memory={pin_memory}"
                )

            return {
                "dataset": ds,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "collate_fn": collate_fn,
                "pin_memory": pin_memory,
                "persistent_workers": True  # workers stay alive across epochs
            }
    
        def _setup_dataloaders(self, args):

            self.testDataLoader = DataLoader(
                **self._adjust_loader_params(args=args)
            )
    
        def prepare(self):
            if not self.P.cnfg:
                print("not initiated")
                return
            # print(self.P.pplid)
            args = deepcopy(self.P.cnfg["args"])
            P = PipeLine(pplid=args['pplid'])
            self.comps["model"] = self.load_component(**P.cnfg['args']['model']).to(self.device) 

            weihts_path = P.get_path(of='weight.weight', args={'epoch':args['epoch']})
            if not os.path.exists(weihts_path):
                raise ValueError(f"weights not available for epoch: {args['epoch']}" )
            state_dict = torch.load(weihts_path)  # load weights
            self.comps["model"].load_state_dict(state_dict) 
            self._setup_dataloaders(args=args)

            print("Data loaders are successfully created")

            return True

        def log(self,of,data):
            if of=='logit.pred':
                torch.save(data, self.P.get_path(of='logit.pred'))
        
        def get_path(self, of, pplid, args):
            path = None
            if of == 'logit.pred':
                path = Path('logit')/ 'pred' / f"{pplid}.pt"
            else:
                raise ValueError(
                    f"Invalid 'of' argument: {of}. Expected 'config', 'result' or 'prediction'."
                )
            return path
        
        def run(self):
            if not os.path.exists(self.P.get_path(of='logit.pred')):
                self.comps["model"].eval()
                all_idxs = []
                all_preds = torch.empty(0).to(self.device)
                with torch.no_grad():
                    for inpts, names in self.testDataLoader:
                        inpts = inpts.to(self.device)
                        _,preds = self.comps["model"](inpts)
                        all_preds = torch.cat([all_preds, preds], dim=0)

                        all_idxs.extend(names)

                predictions = {
                    "names": all_idxs,
                    "preds": all_preds
                }

                self.log(of='logit.pred', data = predictions)

        def new(self,args):
            if not self.template.issubset(set(args.keys())):
                raise ValueError(f'the args should have {", ".join(self.template- set(list(args.keys())))}')
            ppls = get_ppls()
            if args['pplid'] not in ppls:
                raise ValueError(f'invalid pplid: {args["pplid"]}')
            P = PipeLine(pplid = args['pplid'])
            w_path = P.get_path(of='weight.weight', args={'epoch':args['pplid']})
            if os.path.exists(w_path):
                raise ValueError(f'weights for pplid: {args["pplid"]} at epoch: {args["epoch"]} is not available')
    

models
------

.. code-block:: python

    # Model
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from lyf.utils import Model

    class SimpleNN(Model):
        def __init__(self):
            super().__init__()
            self.args = {"h1_dim":None, "h2_dim":None,'drop':None}

        def _setup(self, args):
            h1_dim, h2_dim, drop = args['h1_dim'], args['h2_dim'], args['drop']
            self.seq = nn.Sequential(
                nn.Linear(14, h1_dim),
                nn.ReLU(),
                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(),
                nn.Linear(h2_dim, h2_dim*2),
                nn.ReLU(),
                nn.Linear(h2_dim*2, h2_dim*2),
                nn.ReLU()
            )

            self.dropout = nn.Dropout(p=drop)
            self.final = nn.Linear(h2_dim*2, 1)

        def forward(self, x):
            x = self.seq(x)
            x = self.dropout(x)
            x = self.final(x)
            return x

    class SimpleNNe(Model):
        def __init__(self):
            super().__init__()
            self.args = {"embedding_info":None, "continuous_dim":None,'hidden_dim':None, 'drop':None}

        def _setup(self, args):
            embedding_info, continuous_dim, hidden_dim,drop = args['embedding_info'], args['continuous_dim'], args['hidden_dim'], args['drop']
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_categories, emb_dim)
                for num_categories, emb_dim in embedding_info
            ])

            self.continuous_dim = continuous_dim
            total_emb_dim = sum(emb_dim for _, emb_dim in embedding_info)

            self.fc = nn.Sequential(
                nn.Linear(total_emb_dim + continuous_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x_cat, x_cont):
            x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat(x, dim=1)
            x = torch.cat([x, x_cont], dim=1)
            return self.fc(x)


datasets
--------

.. code-block:: python

    from lyf.utils import DataSet

    import torch

    import pandas as pd
    class DS01(DataSet):
        def __init__(self):
            self.args = {"data_src":None}

        def _setup(self, args):
            self.df = pd.read_csv(args['data_src'])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx, :].values
            row = torch.tensor(row, dtype=torch.float32)  # Convert entire row to float32 tensor
            label = row[-1]
            data = row[:-1]
            return [data], [label]


    import pandas as pd
    import torch
    class DS02(DataSet):
        def __init__(self):
            self.args = {"data_src":None}

        def _setup(self, args):

            self.df = pd.read_csv(args['data_src'])
            self.df.replace('?', pd.NA, inplace=True)
            self.df = self.df.dropna()
            # Define categorical and continuous columns
            self.cat_cols = [
                'workclass', 'education', 'marital_status', 'relationship', 'race',
                'occupation', 'native_country'
            ]
            self.cont_cols = [
                'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
            ]
            self.label_col = 'income'

            # Define mappings for categorical columns (ensure this matches your earlier mappings)
            self.label_encoders = {
                'workclass': {
                    'Private': 0, 'Local-gov': 1, 'Self-emp-not-inc': 2, 'Federal-gov': 3,
                    'State-gov': 4, 'Self-emp-inc': 5, 'Without-pay': 6, 'Never-worked': 7
                },
                'education': {
                    '11th': 0, 'HS-grad': 1, 'Assoc-acdm': 2, 'Some-college': 3, '10th': 4,
                    'Prof-school': 5, '7th-8th': 6, 'Bachelors': 7, 'Masters': 8, '5th-6th': 9,
                    'Assoc-voc': 10, '9th': 11, 'Doctorate': 12, '12th': 13, '1st-4th': 14, 'Preschool': 15
                },
                'marital_status': {
                    'Never-married': 0, 'Married-civ-spouse': 1, 'Widowed': 2,
                    'Divorced': 3, 'Separated': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6
                },
                'relationship': {
                    'Own-child': 0, 'Husband': 1, 'Not-in-family': 2,
                    'Unmarried': 3, 'Wife': 4, 'Other-relative': 5
                },
                'race': {
                    'Black': 0, 'White': 1, 'Other': 2, 'Amer-Indian-Eskimo': 3, 'Asian-Pac-Islander': 4
                },
                'occupation': {
                    'Machine-op-inspct': 0, 'Farming-fishing': 1, 'Protective-serv': 2,
                    'Other-service': 3, 'Prof-specialty': 4, 'Craft-repair': 5,
                    'Adm-clerical': 6, 'Exec-managerial': 7, 'Tech-support': 8,
                    'Sales': 9, 'Priv-house-serv': 10, 'Transport-moving': 11,
                    'Handlers-cleaners': 12, 'Armed-Forces': 13
                },
                'native_country': {
                    'United-States': 0, 'Peru': 1, 'Guatemala': 2, 'Mexico': 3, 'Dominican-Republic': 4,
                    'Ireland': 5, 'Germany': 6, 'Philippines': 7, 'Thailand': 8, 'Haiti': 9, 'El-Salvador': 10,
                    'Puerto-Rico': 11, 'Vietnam': 12, 'South': 13, 'Columbia': 14, 'Japan': 15, 'India': 16,
                    'Cambodia': 17, 'Poland': 18, 'Laos': 19, 'England': 20, 'Cuba': 21, 'Taiwan': 22,
                    'Italy': 23, 'Canada': 24, 'Portugal': 25, 'China': 26, 'Nicaragua': 27, 'Honduras': 28,
                    'Iran': 29, 'Scotland': 30, 'Jamaica': 31, 'Ecuador': 32, 'Yugoslavia': 33, 'Hungary': 34,
                    'Hong': 35, 'Greece': 36, 'Trinadad&Tobago': 37, 'Outlying-US(Guam-USVI-etc)': 38,
                    'France': 39, 'Holand-Netherlands': 40
                }
            }

            # Encode categorical variables
            for col, mapping in self.label_encoders.items():
                self.df[col] = self.df[col].replace(mapping)

            # Encode label column
            self.df[self.label_col] = self.df[self.label_col].replace({'<=50K': 0, '>50K': 1})

            # Convert everything to torch tensors
            self.cat_data = torch.tensor(self.df[self.cat_cols].values, dtype=torch.long)
            self.cont_data = torch.tensor(self.df[self.cont_cols].values, dtype=torch.float32)
            self.labels = torch.tensor(self.df[self.label_col].values, dtype=torch.float32).unsqueeze(1)
        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            label = self.labels[idx]
            return [self.cat_data[idx], self.cont_data[idx]], [label]

losses
------

.. code-block:: python
    #Loss
    from lyf.utils import Loss
    from torch import nn

    class BCElogit(Loss):
        def __init__(self):
            super().__init__()
            self.args ={}
        def _setup(self,args):
            self.criterion = nn.BCEWithLogitsLoss()

        def forward(self, logits, y_true):
            # y_true = y_true[0]
            # print(16, logits.shape,y_true.shape)
            logits = logits.view_as(y_true)
            loss = self.criterion(logits, y_true.float())
            # print(16, logits.shape,y_true.shape, loss)
            return loss


optimizers
----------

.. code-block:: python

    from lyf.utils import Optimizer
    import torch.optim as optim

    class OptAdam(Optimizer):
        def __init__(self):
            super().__init__()
            self.optimizer = None

        def _setup(self,args):
            learning_rate = args.get('learning_rate', 0.001)
            self.optimizer = optim.Adam(args['model_parameters'], lr=learning_rate)
        def step(self, **kwargs):
            # Step function to apply the gradients and update model parameters
            self.optimizer.step()

        def zero_grad(self):
            # Zero the gradients before the backward pass
            self.optimizer.zero_grad()


metrics
--------

.. code-block:: python

    # metrics

    import torch
    from lyf.utils import Metric
    from torchmetrics.classification import BinaryAccuracy

    class BinAcc(Metric):
        def __init__(self):
            super().__init__()
            # self.args = {'threshold':None}

        def _setup(self, args):
            thres = args.get('threshold',0.5)
            self.accuracy = BinaryAccuracy(threshold=thres)

        def forward(self,y_pred, y_true):
            y_pred = y_pred.view_as(y_true)
            accuracy = self.accuracy(y_pred, y_true)
            return accuracy.item()


    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    class AUROC(Metric):
        def __init__(self):
            super().__init__()
        def _setup(self, args):
            pass
        def forward(self, outputs, targets):
            if outputs.size(1) == 1:
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                # print(targets.shape, probabilities.shape)
                auroc = roc_auc_score(targets, probabilities)
            # For multi-class classification (softmax)
            else:
                probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                # One-hot encode targets for multi-class
                auroc = roc_auc_score(targets, probabilities, average='macro', multi_class='ovr')

            return auroc

    from sklearn.metrics import average_precision_score

    class AUPRC(Metric):
        def __init__(self):
            super().__init__()
        def _setup(self, args):
            pass
        def forward(self, outputs, targets):
            if outputs.size(1) == 1:
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                auprc = average_precision_score(targets, probabilities)
            # For multi-class classification (softmax)
            else:
                probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                # For multi-class, use average_precision_score for each class separately and average
                auprc = average_precision_score(targets, probabilities, average='macro', multi_class='ovr')

            return auprc

    from sklearn.metrics import f1_score
    class F1Score(Metric):
        def __init__(self):
            super().__init__()
        def _setup(self, args):
                pass
        def forward(self, outputs, targets):
            if outputs.size(1) == 1:
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)  # Convert to 0 or 1 (binary classification)
                targets = targets.detach().cpu().numpy()
                f1 = f1_score(targets, predictions)
            # For multi-class classification (softmax)
            else:
                probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                predictions = probabilities.argmax(axis=1)  # Choose the class with the highest probability
                targets = targets.detach().cpu().numpy()
                f1 = f1_score(targets, predictions, average='macro')  # Macro-average for multi-class
            return f1



Other Components
----------------

.. code-block:: python

    from lwf.utils import Component
    class NullEarlyStopper(Component):
        def __init__(self):
            super().__init__()        

        def _setup(self, args):
            self.early_stop = False  # Never triggers
        def __call__(self, data):
            # Always do nothing and never stop
            self.early_stop = False
        def state_dict(self):
            return {
            }
        def load_state(self, state):
            pass
        def step(self, data):
            pass

    import torch
    class EarlyStopper(Component):
        def __init__(self):
            super().__init__()
            self.args = {"patience", "min_delta", "mode", 'monitor'}

        def _setup(self,args):
            self.patience = args["patience"]#5
            self.min_delta = args["min_delta"]#0
            self.counter = 0
            self.monitor = args['monitor']
            self.best_score = None
            self.early_stop = False
            self.mode = args["mode"]#'min'

            if self.mode == 'min':
                self.monitor_op = lambda current, best: current < best - self.min_delta
            elif self.mode == 'max':
                self.monitor_op = lambda current, best: current > best + self.min_delta
            else:
                raise ValueError("mode must be 'min' or 'max'")

        def step(self, data):
            current_score = data[self.monitor]
            if self.best_score is None:
                self.best_score = current_score
            elif not self.monitor_op(current_score, self.best_score):
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        
        def state_dict(self):
            return {
                "best_score": self.best_score,
                "counter": self.counter,
                "early_stop": self.early_stop
            }

        def load_state(self, state):
            self.best_score = state.get("best_score", None)
            self.counter = state.get("counter", 0)
            self.early_stop = state.get("early_stop", False)

    class LRScheduler(Component):
        def __init__(self):
            super().__init__()
            self.args = {"mode", "monitor", 'optimizer', 'factor', 'patience'}
        
        def _setup(self, args):
            self.mode = args.get("mode")  # 'step' or 'metric'
            self.monitor = args.get("monitor")  # e.g., "val_loss"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                args['optimizer'],
                mode=args['mode'],
                factor=args['factor'],
                patience=args['patience']
            )
            self.scheduler = scheduler
        def step(self, data=None):
            metric = data[self.monitor]
            self.scheduler.step(metric)

        def get_last_lr(self):
            return self.scheduler.get_last_lr()

        def state_dict(self):
            return self.scheduler.state_dict()

        def load_state_dict(self, state):
            self.scheduler.load_state_dict(state)

    class NullScheduler(Component):
        def __init__(self):
            super().__init__()
        def _setup(self, args):
            pass
        def step(self, data=None):
            pass

        def get_last_lr(self):
            return []

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass


