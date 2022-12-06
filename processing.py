from rich.progress import track
from dataset import DatasetForFeaturesExtraction
from libraries.log import logger
from libraries.strategies import * 
import argparse

def processing(path2vectorizer, path2images, path2captions, extension, path2features, path2tokenids, path2vocabulary):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    with open(file=path2captions, mode='r') as fp:
        img2captions = json.load(fp)
    
    captions = list(img2captions.values())
    captions = list(it.chain(*captions))

    tokenizer = build_tokenizer(tok_name='spacy', lang='en_core_web_sm')
    vocabulary = make_vocab(captions, tokenizer, SPECIALS2IDX)
    logger.success('vocaulary was built')
    
    serialize(path2vocabulary, vocabulary)

    bos = th.tensor([SPECIALS2IDX['<bos>']])
    eos = th.tensor([SPECIALS2IDX['<eos>']])
        
    zip_img2tokenids = []
    logger.debug('caption tokenization')
    for key, val in track(img2captions.items(), 'build map_img2tokenids'):
        for cap in val:
            tok = tokenizer(cap.strip().lower())
            idx = th.tensor(vocabulary(tok))
            seq = th.cat([bos, idx, eos]).numpy()  # more effective for storage 
            zip_img2tokenids.append((key, seq))
    
    serialize(path2tokenids, zip_img2tokenids)

    logger.debug('features extraction loading')
    vectorizer = load_vectorizer(path2vectorizer)
    vectorizer.eval()
    vectorizer.to(device)

    dataset = DatasetForFeaturesExtraction(path2images, f'*.{extension}')

    logger.debug('extraction will start')
    accumulator = []
    for sections in track(dataset, 'features extraction'):
        embedding = extract_features(vectorizer, sections[None, ...].to(device)).squeeze(0) # (2048, 7, 7)
        embedding = th.flatten(embedding, start_dim=1).T.cpu().numpy()  # 49, 2048
        accumulator.append(embedding)
    
    image_names = dataset.image_names
    accumulator = np.stack(accumulator)  # stack over batch axis ==> (nb_images, 49, 512)
    logger.debug(f'accumulated features shape : {accumulator.shape}')
    assert len(image_names) == len(accumulator)
    map_img2features = dict(zip(image_names, accumulator)) 

    serialize(path2features, map_img2features)

    logger.success('features, tokenids and vocabulary were saved')


def main(args):
    processing(args.vectorizer, args.images, args.captions, args.image_ext, args.features, args.tokenids, args.vocabulary)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # add necessary arguments to parser
    parser.add_argument('--vectorizer', type=str, default='src/resnet152.th')
    parser.add_argument('--images', type=str, default='dataset/flickr30k_images')
    parser.add_argument('--captions', type=str, default='dataset/captions.json')
    parser.add_argument('--image_ext', type=str, default='jpg')
    parser.add_argument('--features', type=str, default='src/map_img2features.pkl')
    parser.add_argument('--tokenids', type=str, default='src/zip_img2tokenids.pkl')
    parser.add_argument('--vocabulary', type=str, default='src/vocabulary.pkl')
    parser.add_argument('--checkpoint', type=str, default='src/checkpoint_###.th')
    
    args = parser.parse_args()
    main(args)