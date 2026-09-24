import torch
from tqdm import tqdm
import gc

def generate_unique_index(start, end, length):
    return torch.randperm(end - start)[:length].tolist()

def get_cos_similar_matrix(v1, v2, device):
    v1 = v1.to(device)
    v2 = v2.to(device)
    num = torch.mm(v1, v2.t())
    denom = torch.norm(v1, dim=1).reshape(-1, 1) * torch.norm(v2, dim=1)
    res = num / denom
    res[torch.isinf(res)] = 0
    res = 0.5 + 0.5 * res
    res = res.cpu()

    del v1, v2, num, denom
    return res

def average_similarity(layer_cosine_similarity):
    return torch.tensor(layer_cosine_similarity).mean().item()

@torch.no_grad() 
def get_cosine_similarity(model, dataset, num_data, device, layer_intervals, num_layer):
    assert len(dataset) > num_data
    model = model.to(device)
    hidden_states_list = []
    data_index = generate_unique_index(0, len(dataset), num_data)
    
    for i in tqdm(data_index, desc="Collecting hidden states"):
        input_ids = torch.tensor(dataset[i]['input_ids'])
            
        if len(input_ids.shape) != 2:
            input_ids = input_ids.reshape(1, -1)
        input_ids = input_ids.to(device)
        
        hidden_states = model(input_ids, output_hidden_states=True).hidden_states
        hidden_states = [h.cpu() for h in hidden_states]
        hidden_states_list.append(hidden_states)
        
        del input_ids
    
    cosine_similarity = [[] for _ in range(num_layer - layer_intervals + 1)]
    
    for i in range(len(hidden_states_list)):
        for j in range(num_layer - layer_intervals + 1):
            cosine = get_cos_similar_matrix(
                hidden_states_list[i][j][0], 
                hidden_states_list[i][j+layer_intervals][0],
                device
            )
            similarity = torch.trace(cosine) / cosine.size(0)
            cosine_similarity[j].append(similarity.item())
            del cosine
    
    print('Calculating cosine similarity...')
    similarities = [average_similarity(layer_sim) for layer_sim in cosine_similarity]
    similarities_tensor = torch.tensor(similarities)
    best_layer = torch.argmax(similarities_tensor).item()
    best_cosine = similarities[best_layer]
    
    for i, sim in enumerate(similarities):
        print(f'The cosine similarity between hidden_states {i} and hidden_states {i + layer_intervals} is {sim:.4f}')
    
    print(f'The highest cosine similarity comes from hidden_states {best_layer} and hidden_states {best_layer + layer_intervals}, with a value of {best_cosine:.4f}')

    model.cpu()
    del hidden_states_list, model
    torch.cuda.empty_cache()  
    
    return best_layer