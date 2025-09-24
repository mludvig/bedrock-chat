import { Trans, useTranslation } from 'react-i18next';
import {
  AgentTool,
  BedrockAgentConfig,
  BedrockAgentTool,
  InternetSearchConfig,
  InternetAgentTool,
  ToolType,
} from '../types';
import { isInternetTool, isBedrockAgentTool } from '../utils/typeGuards';
import Toggle from '../../../components/Toggle';
import { Dispatch, useCallback } from 'react';
import { formatDescription } from '../functions/formatDescription';
import Help from '../../../components/Help';
import Skeleton from '../../../components/Skeleton';
import { TooltipDirection } from '../../../constants';
import { InternetSearchConfig as InternetSearchConfigComponent } from './InternetSearchConfig';
import { BedrockAgentConfig as BedrockAgentConfigComponent } from './BedrockAgentConfig';

import { DEFAULT_INTERNET_SEARCH_CONFIG } from '../constants';

type Props = {
  availableTools: AgentTool[] | undefined;
  tools: AgentTool[];
  setTools: Dispatch<React.SetStateAction<AgentTool[]>>;
};

export const AvailableTools = ({ availableTools, tools, setTools }: Props) => {
  const { t } = useTranslation();

  const handleChangeTool = useCallback(
    (tool: AgentTool) => () => {
      if (tool.name === 'internet_search') {
        setTools((preTools) => {
          const isEnabled = preTools
            ?.map(({ name }) => name)
            .includes(tool.name);

          const newTools = isEnabled
            ? [...preTools.filter(({ name }) => name != tool.name)]
            : [
                ...preTools,
                {
                  ...tool,
                  toolType: 'internet' as ToolType,
                  name: 'internet_search',
                  internetSearchConfig: {
                    maxResults: DEFAULT_INTERNET_SEARCH_CONFIG.maxResults,
                  },
                } as AgentTool,
              ];

          return newTools;
        });
      } else if (tool.name === 'bedrock_agent') {
        setTools((preTools) => {
          const isEnabled = preTools
            ?.map(({ name }) => name)
            .includes(tool.name);

          const newTools = isEnabled
            ? [...preTools.filter(({ name }) => name != tool.name)]
            : [
                ...preTools,
                {
                  ...tool,
                  toolType: 'bedrock_agent' as ToolType,
                  name: 'bedrock_agent',
                  bedrockAgentConfig: {
                    agentId: '',
                    aliasId: '',
                  },
                } as AgentTool,
              ];

          return newTools;
        });
      } else {
        setTools((preTools) =>
          preTools?.map(({ name }) => name).includes(tool.name)
            ? [...preTools.filter(({ name }) => name != tool.name)]
            : [...preTools, tool]
        );
      }
    },
    [setTools]
  );

  const handleInternetSearchConfigChange = useCallback(
    (config: InternetSearchConfig) => {
      setTools((prevTools) =>
        prevTools.map((tool) => {
          if (tool.name === 'internet_search') {
            return {
              ...tool,
              toolType: 'internet' as ToolType,
              name: 'internet_search',
              internetSearchConfig: config,
            } as AgentTool;
          }
          return tool;
        })
      );
    },
    [setTools]
  );

  const handleBedrockAgentConfigChange = useCallback(
    (config: BedrockAgentConfig) => {
      setTools((prevTools) =>
        prevTools.map((tool) => {
          if (tool.name === 'bedrock_agent') {
            return {
              ...tool,
              toolType: 'bedrock_agent' as ToolType,
              name: 'bedrock_agent',
              bedrockAgentConfig: config,
            } as AgentTool;
          }
          return tool;
        })
      );
    },
    [setTools]
  );



  return (
    <>
      <div className="flex items-center gap-1">
        <div className="text-lg font-bold dark:text-aws-font-color-dark">
          {t('agent.label')}
        </div>
        <Help direction={TooltipDirection.RIGHT} message={t('agent.hint')} />
      </div>

      <div className="text-sm text-aws-font-color-light/50 dark:text-aws-font-color-dark">
        <Trans
          i18nKey="agent.help.overview"
          components={{
            Link: (
              <a
                href="https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html"
                target="_blank"
                rel="noopener noreferrer"
                className="text-aws-sea-blue-light underline dark:text-aws-sea-blue-dark"
              />
            ),
          }}
        />
      </div>
      {availableTools === undefined && <Skeleton className="h-12 w-full" />}

      {availableTools?.map((tool) => (
        <div key={tool.name}>
          <div className="flex items-center">
            <Toggle
              value={!!tools?.map(({ name }) => name).includes(tool.name)}
              onChange={handleChangeTool(tool)}
            />
            <div className="whitespace-pre-wrap text-sm text-aws-font-color-light/50 dark:text-aws-font-color-dark">
              {formatDescription(tool, t)}
            </div>
          </div>
          {tool.name === 'internet_search' &&
            tools?.map(({ name }) => name).includes('internet_search') && (
              <div className="ml-8 mt-2">
                <InternetSearchConfigComponent
                  config={
                    tools.find(
                      (t): t is InternetAgentTool =>
                        t.name === 'internet_search' &&
                        isInternetTool(t)
                    )?.internetSearchConfig || {
                      maxResults: DEFAULT_INTERNET_SEARCH_CONFIG.maxResults,
                    }
                  }
                  onChange={handleInternetSearchConfigChange}
                />
              </div>
            )}
          {tool.name === 'bedrock_agent' &&
            tools?.map(({ name }) => name).includes('bedrock_agent') && (
              <div className="space-y-4">
                <div className="ml-6 text-sm">
                  <BedrockAgentConfigComponent
                    config={
                      tools.find(
                        (t): t is BedrockAgentTool =>
                          t.name === 'bedrock_agent' && isBedrockAgentTool(t)
                      )?.bedrockAgentConfig || {
                        agentId: '',
                        aliasId: '',
                      }
                    }
                    onChange={handleBedrockAgentConfigChange}
                  />
                </div>
              </div>
            )}
        </div>
      ))}
    </>
  );
};
