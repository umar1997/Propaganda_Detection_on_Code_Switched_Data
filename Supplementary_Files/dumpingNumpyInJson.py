json_dump= json.dumps(test_dict, cls=NumpyEncoder)
        with open("test_split.json", "w") as f:
            json.dump(json_dump, f)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
