import configparser
import os
from typing import Any, Dict


# A mapping from parameters name -> final type
_params = {
    # 图像参数
    'Graphics': {
        'tile_size': (tuple, float),
        'neuron_radius': float,
    },

    # 统计参数
    'Statistics': {
        'save_best_individual_from_generation': str,
        'save_population_stats': str,
    },

    # 神经网络参数
    'NeuralNetwork': {
        'input_dims': (tuple, int),
        'hidden_layer_architecture': (tuple, int),
        'hidden_node_activation': str,
        'output_node_activation': str,
        'encode_row': bool,
    },

    # 遗传算法
    'GeneticAlgorithm': {
        'fitness_func': type(lambda : None)
    },

    # 重组参数
    'Crossover': {
        'probability_sbx': float,
        'sbx_eta': float,
        'crossover_selection': str,
        'tournament_size': int,
    },

    # 变异参数
    'Mutation': {
        'mutation_rate': float,
        'mutation_rate_type': str,
        'gaussian_mutation_scale': float,
    },

    # 选择参数
    'Selection': {
        'num_parents': int,
        'num_offspring': int,
        'selection_type': str,
        'lifespan': float
    },

    # 其他参数
    'Misc': {
        'level': str,
        'allow_additional_time_for_flagpole': bool,
        'max_static_duration': int,
        'fps': int
    }
}

# 展开多重字典为二重字典
class DotNotation(object):
    def __init__(self, d: Dict[Any, Any]):
        for k in d:
            # If the key is another dictionary, keep going
            if isinstance(d[k], dict):
                self.__dict__[k] = DotNotation(d[k])
            # If it's a list or tuple then check to see if any element is a dictionary
            elif isinstance(d[k], (list, tuple)):
                l = []
                for v in d[k]:
                    if isinstance(v, dict):
                        l.append(DotNotation(v))
                    else:
                        l.append(v)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = d[k]
    
    def __getitem__(self, name) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)

#整个程序的配置
class Config(object):
    def __init__(self, filename: str):
        if not '.config' in filename:
            self.filename = os.path.join(filename, 'settings.config')
        else:
            self.filename = filename
        
        if not os.path.isfile(self.filename):
            raise Exception('No file found named "{}"'.format(self.filename))

        with open(self.filename, errors='ignore') as f:
            self._config_text_file = f.read()

        self._config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._config.read(self.filename,encoding="utf-8-sig") # 这里将配置文件载入配置

        self._verify_sections()
        self._create_dict_from_config()
        self._set_dict_types()
        dot_notation = DotNotation(self._config_dict)
        self.__dict__.update(dot_notation.__dict__)

    # 从配置中生成字典
    def _create_dict_from_config(self) -> None:
        d = {}
        for section in self._config.sections():
            d[section] = {}
            for k, v in self._config[section].items():
                d[section][k] = v
        self._config_dict = d
        
    # 多余字段检测(这不和下面的重复了？)，并且将配置转换为对应类型
    def _set_dict_types(self) -> None:
        for section in self._config_dict:
            for k, v in self._config_dict[section].items():
                try:
                    _type = _params[section][k]
                except:
                    raise Exception('No value "{}" found for section "{}". Please set this in _params'.format(k, section))
                # Normally _type will be int, str, float or some type of built-in type.
                # 如果类型是元组，则我们需要切分数据
                if isinstance(_type, tuple):
                    if len(_type) == 2:
                        cast = _type[1]
                        v = v.replace('(', '').replace(')', '')  # Remove any parens that might be present 
                        self._config_dict[section][k] = tuple(cast(val) for val in v.split(','))
                    else:
                        raise Exception('Expected a 2 tuple value describing that it is to be parse as a tuple and the type to cast it as')
                # 检测是否是lambda表达式
                elif 'lambda' in v:
                    try:
                        self._config_dict[section][k] = eval(v)
                    except:
                        print("生存适应度的评价算法lambda有误！已跳过")
                        pass
                # 检测是否是布尔类型
                elif _type == bool:
                    self._config_dict[section][k] = _type(eval(v))
                # Otherwise parse normally
                else:
                    self._config_dict[section][k] = _type(v)

    # 验证有无多余字段
    def _verify_sections(self) -> None:
        # Validate sections
        for section in self._config.sections():
            # Make sure the section is allowed
            if section not in _params:
                raise Exception('Section "{}" has no parameters allowed. Please remove this section and run again.'.format(section))

    def _get_reference_from_dict(self, reference: str) -> Any:
        path = reference.split('.')
        d = self._config_dict
        for p in path:
            d = d[p]
        
        assert type(d) in (tuple, int, float, bool, str)
        return d

    def _is_number(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False
