#!/usr/bin/env python3
"""
Convertir workflow ComfyUI du format UI au format API
"""
import json
import sys

def convert_ui_to_api(ui_workflow):
    """Convertit le format UI ComfyUI en format API"""

    api_workflow = {}

    # Map node_id → node data
    nodes_map = {str(node['id']): node for node in ui_workflow['nodes']}

    # Map link_id → link data
    links_map = {link[0]: link for link in ui_workflow.get('links', [])}

    # Pour chaque nœud
    for node in ui_workflow['nodes']:
        node_id = str(node['id'])
        inputs = {}

        # Process inputs (connections from other nodes)
        if 'inputs' in node:
            for inp in node['inputs']:
                input_name = inp['name']

                # Si c'est connecté à un autre nœud
                if 'link' in inp and inp['link'] is not None:
                    link_id = inp['link']
                    if link_id in links_map:
                        link = links_map[link_id]
                        # link format: [link_id, source_node_id, source_slot, target_node_id, target_slot, type]
                        source_node_id = str(link[1])
                        source_slot = link[2]
                        inputs[input_name] = [source_node_id, source_slot]

        # Process widgets_values
        if 'widgets_values' in node and node['widgets_values']:
            # Map widgets to their input names
            # Note: This is simplified - real mapping depends on node type
            widget_values = node['widgets_values']

            # For most nodes, widgets map to inputs by order
            # This is a basic implementation
            if isinstance(widget_values, list):
                # Try to find widget names from node properties
                if node['type'] == 'CLIPTextEncode':
                    inputs['text'] = widget_values[0] if widget_values else ""
                elif node['type'] == 'CheckpointLoaderSimple':
                    inputs['ckpt_name'] = widget_values[0] if widget_values else ""
                elif node['type'] == 'LoadImage':
                    inputs['image'] = widget_values[0] if widget_values else ""
                elif node['type'] == 'LoraLoaderModelOnly':
                    if len(widget_values) >= 2:
                        inputs['filename'] = widget_values[0]
                        inputs['strength_model'] = widget_values[1]
                elif node['type'] == 'LTXAVTextEncoderLoader':
                    if len(widget_values) >= 3:
                        inputs['filename'] = widget_values[0]
                        inputs['ckpt_name'] = widget_values[1]
                        inputs['device'] = widget_values[2]
                elif node['type'] == 'ResizeImageMaskNode':
                    if len(widget_values) >= 2:
                        inputs['resize_mode'] = widget_values[0]
                        inputs['scale_factor'] = widget_values[1]
                        if len(widget_values) >= 3:
                            inputs['interpolation'] = widget_values[2]
                elif node['type'] == 'EmptyLTXVLatentVideo':
                    if len(widget_values) >= 3:
                        inputs['width'] = widget_values[0]
                        inputs['height'] = widget_values[1]
                        inputs['length'] = widget_values[2]
                        if len(widget_values) >= 4:
                            inputs['batch_size'] = widget_values[3]
                elif node['type'] == 'VHS_VideoCombine':
                    if len(widget_values) >= 2:
                        inputs['frame_rate'] = widget_values[0]
                        inputs['loop_count'] = widget_values[1]
                        if len(widget_values) >= 3:
                            inputs['filename_prefix'] = widget_values[2]
                        if len(widget_values) >= 4:
                            inputs['format'] = widget_values[3]
                # Add more node type mappings as needed

        # Build API node
        api_node = {
            "inputs": inputs,
            "class_type": node['type']
        }

        # Add metadata if present
        if 'title' in node and node['title']:
            api_node["_meta"] = {"title": node['title']}

        api_workflow[node_id] = api_node

    return api_workflow

def main():
    if len(sys.argv) < 2:
        print("Usage: ./convert_ui_to_api.py workflow-ui.json > workflow-api.json")
        sys.exit(1)

    # Lire le workflow UI
    with open(sys.argv[1], 'r') as f:
        ui_workflow = json.load(f)

    # Convertir
    api_workflow = convert_ui_to_api(ui_workflow)

    # Sortie
    output = {
        "input": {
            "workflow": api_workflow
        }
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
