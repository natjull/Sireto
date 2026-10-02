from scripts.retrieval_v39_rne_projection import project


def test_documented_rne_fields_survive_with_scope_and_dates():
    power=dict(typeDePersonne='ENTREPRISE',roleEntreprise='99',dateEffet='2020-02-01',
        entreprise=dict(siren='222222222',denomination='CABINET OFFICIEL',libelleRoleEntreprise='Rôle officiel'),
        etablissementConcerne=[dict(siret='11111111100001',dateEffetEtablissementConcerne='2020-03-01')])
    raw=dict(id='official1',siren='111111111',content=dict(personneMorale=dict(composition=dict(pouvoirs=[power]))))
    value,=project(raw)
    assert value['official_role_label']=='Rôle officiel'
    assert value['role_effective_date']=='2020-02-01'
    assert value['site_evidence']==[dict(siret='11111111100001',date='2020-03-01',source_field='etablissementConcerne')]
    assert value['active'] is None
    assert value['navigation'].startswith('CONTEXT_ONLY')
    assert 'actif' not in power  # source record is never mutated


def test_individuals_are_ignored_and_missing_related_id_does_not_invent_link():
    powers=[dict(typeDePersonne='INDIVIDU',individu=dict(nom='Never project')),
        dict(entreprise=dict(denomination='PERSONNE MORALE SANS IDENTIFIANT'),roleEntreprise='?')]
    raw=dict(id='official2',siren='111111111',content=dict(personneMorale=dict(composition=dict(pouvoirs=powers))))
    value,=project(raw)
    assert value['related_siren']=='' and value['role_code']=='?'
    assert not value['site_evidence'] and value['navigation'].startswith('CONTEXT_ONLY')
