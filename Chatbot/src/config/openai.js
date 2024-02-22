import OpenAI from 'openai';
const openai = new OpenAI({
  apiKey: process.env['OPEN_AI_API_KEY'],
  dangerouslyAllowBrowser: true,
});

export default openai;
