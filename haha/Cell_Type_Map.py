import time
import tangram as tg
import pandas as pd
import anndata as ad
import os


def get_anndata_structure(data):
    adata = ad.AnnData(X=data.to_numpy(), obs=pd.DataFrame(index=data.index.tolist()),
                       var=pd.DataFrame(index=data.columns.values.tolist()))
    return adata


def cell_type_map(data_path, output_path, sc_data_path, device):
    device = "cuda:1" if device == "GPU" else "cpu"
    star_time = time.time()

    all_section = os.listdir(data_path)
    st_data = pd.DataFrame()

    for item in all_section:
        gem = pd.read_csv(os.path.join(output_path, item, "2_gem", "filtered_GEM.csv"), sep=",", header=0, index_col=0).T
        gem["section"] = item
        st_data = pd.concat([st_data, gem])

    sc_data = pd.read_csv(sc_data_path, sep=",", header=0, index_col=0)
    sc_data_cell_type_label = sc_data["cell_type"].tolist()
    sc_data.drop("cell_type", axis=1, inplace=True)

    # get shared gene
    sc_gene_name = sc_data.columns.values.tolist()
    st_gene_name = st_data.columns.values.tolist()
    share_gene = set(sc_gene_name) & set(st_gene_name)

    sc_ad_data = get_anndata_structure(sc_data)
    sc_ad_data.obs["cell_type"] = sc_data_cell_type_label

    st_ad_data = get_anndata_structure(st_data)
    st_ad_data.obs["section"] = st_data["section"]

    tg.pp_adatas(sc_ad_data, st_ad_data, genes=share_gene)

    ad_map = tg.map_cells_to_space(
        sc_ad_data,
        st_ad_data,
        mode='clusters',
        device=device,
        cluster_label='cell_type')

    cluster = ad_map.obs["cell_type"].tolist()

    # 取出预测的概率
    st_predict_prob = pd.DataFrame(ad_map.X).T
    st_predict_prob.rename(columns={i: cluster[i] for i in range(0, len(cluster))}, inplace=True)

    final_result = pd.DataFrame()
    final_result["cell_index"] = ad_map.var.index
    final_result["cell_type"] = st_predict_prob.idxmax(axis=1)
    final_result["section"] = ad_map.var["section"].tolist()

    for item_1 in all_section:
        save_path = os.path.join(output_path, item_1, "5_cell_type_result")
        os.makedirs(save_path, exist_ok=True)
        single_setion = final_result[final_result["section"] == item_1]
        single_setion.drop("section", axis=1, inplace=True)
        final_result.to_csv(os.path.join(save_path, "cell_type.csv"), sep=",", header=True, index=False)

    all_section_result_path = os.path.join(output_path, "all_section_result")
    os.makedirs(all_section_result_path, exist_ok=True)
    final_result.to_csv(os.path.join(all_section_result_path, "cell_type.csv"), sep=",", header=True)
    print("Cell type map finished, runtime：", time.time() - star_time)
