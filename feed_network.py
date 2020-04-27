import argparse
### TODO: Load the necessary libraries
import os
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    c_desc = "CPU extension file location, if applicable"
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-c", help=c_desc, default=CPU_EXTENSION)
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml, cpu_ext):
    ### TODO: Load the Inference Engine API
    plugin = IECore()

    ### TODO: Load IR files into their related class
    bin = os.path.splitext(model_xml)[0] + '.bin'
    network = IENetwork(model=model_xml, weights=bin)

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    # plugin.add_extension(cpu_ext, "CPU")

    ### TODO: Get the supported layers of the network
    supported_layers = plugin.query_network(network=network, device_name="CPU")
    unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    if len(unsupported_layers):
        print(f"Unsupported layers: {unsupported_layers}")
        exit(1)

    ### TODO: Load the network into the Inference Engine
    plugin.load_network(network, "CPU")

    print("IR successfully loaded into Inference Engine.")

    return


def main():
    args = get_args()
    load_to_IE(args.m, args.c)


if __name__ == "__main__":
    main()
