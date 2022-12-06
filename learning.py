from torch.utils.data import DataLoader

from os import getenv, path 
from time import sleep 
from dataset import DatasetForTraining
from libraries.log import logger
from libraries.strategies import * 

from model import CaptionTransformer

import argparse

def learning(path2vocabulary, path2features, path2tokenids, nb_epochs, bt_size, path2checkpoint, checkpoint, start):
    basepath2models = getenv('MODELS')

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    logger.debug('load vocabulary')
    vocabulary = deserialize(path2vocabulary)
    nb_tokens = len(vocabulary)

    logger.debug('build dataset')
    dataset = DatasetForTraining(path2tokenids, path2features)
    logger.debug(f'size of the dataset : {len(dataset):05d}')
    dataloader = DataLoader(dataset, batch_size=bt_size, shuffle=True, collate_fn=custom_fn)
    nb_data = len(dataset)

    logger.debug('define network')
    if path.isfile(path2checkpoint):
        net = th.load(path2checkpoint)
    else:
        net = CaptionTransformer(
            in_dim=2048,
            hd_dim=256,
            ff_dim=512,
            nb_heads=8,
            num_encoders=5,
            num_decoders=5,
            pre_norm=False,
            seq_length=64,
            nb_tokens=nb_tokens,
            padding_idx=SPECIALS2IDX['<pad>'] 
        )
    
    net.to(device)
    net.train()
    
    print(net)

    optimizer = th.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIALS2IDX['<pad>'])
    logger.debug('training  will begin ...!')
    sleep(1)

    nb_epochs += start 
    for epoch in range(start, nb_epochs):
        counter = 0
        for src, tgt in dataloader:
            counter += len(tgt)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = build_mask(tgt_input).to(device)
            tgt_key_padding_mask = build_key_padding_mask(tgt_input, SPECIALS2IDX['<pad>']).to(device)
            
            memory = net.encode(src=src.to(device))
            output = net.decode(
                tgt=tgt_input.to(device), 
                memory=memory, 
                tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            logits = [net.generator(out) for out in output ]
            logits = [ th.flatten(prb, start_dim=0, end_dim=1) for prb in logits ]
            tgt_output = th.flatten(tgt_output)

            optimizer.zero_grad() 
            errors = [ criterion(prb, tgt_output.to(device)) for prb in logits ]
            error = sum(errors)
            error.backward()
            optimizer.step()

            message = []
            for err in errors:
                msg = f'{err.cpu().item():07.3f}'
                message.append(msg)
            message = ' | '.join(message)
            logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{counter:05d}/{nb_data:05d}] | Loss : {error.cpu().item():07.3f} >> {message}')
        # end for loop over batchs 
        
        if epoch % checkpoint == 0:
            path2network = path.join(basepath2models, f'checkpoint_{epoch:03d}.th')
            th.save(net.cpu(), path2network)
            net.to(device)
            logger.success(f'a snapshot was saved {path2network}')

    # end for loop over epochs 
    
    path2network = path.join(basepath2models, f'checkpoint_###.th')
    th.save(net.cpu(), path2network)
    logger.success(f'a snapshot was saved {path2network}')
    logger.success('end of training')


def main(args):
    learning(args.vocabulary, args.features, args.tokenids, args.nb_epochs, args.bt_size, args.checkpoint_path, args.checkpoint, args.start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='src/map_img2features.pkl')
    parser.add_argument('--tokenids', type=str, default='src/zip_img2tokenids.pkl')
    parser.add_argument('--vocabulary', type=str, default='src/vocabulary.pkl')
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--bt_size', type=int, default=128)
    parser.add_argument('--checkpoint_path', type=str, default='models/checkpoint_128.th')
    parser.add_argument('--checkpoint', type=int, default=16)
    parser.add_argument('--start', type=int, default=0)
    
    args = parser.parse_args()
    main(args)